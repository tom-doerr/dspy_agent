import dspy
import json
from rich.console import Console
from .rating import RatingModule
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
import xml.etree.ElementTree as ET
from lxml import etree
import io
from .schema import (
    INPUT_XML_SCHEMA,
    OUTPUT_XML_SCHEMA,
    PLAN_XML_SCHEMA,
    EXECUTION_XML_SCHEMA,
    EXAMPLE_INPUT_XML,
    EXAMPLE_OUTPUT_XML,
    EXAMPLE_PLAN_XML,
    EXAMPLE_EXECUTION_XML
)

class UnifiedTask(dspy.Signature):
    """Generate output XML with updated memory, new plan, and execution instructions from input XML."""
    input_xml = dspy.InputField(desc=f"Input XML with memory, last_plan, last_action, observation. Schema: {INPUT_XML_SCHEMA}")
    output_xml = dspy.OutputField(desc=f"""Output XML with:
    - updated_memory: Updated knowledge based on observations
    - new_plan: A structured plan following this schema: {PLAN_XML_SCHEMA}
    - execution_instructions: Write operations to execute following this schema: {EXECUTION_XML_SCHEMA}
    - is_done: Boolean indicating if the task is complete
    
    Full schema: {OUTPUT_XML_SCHEMA}""")

class UnifiedModule(dspy.Module):
    def __init__(self, teleprompter=None, optimizer: str = "bootstrap"):
        super().__init__()
        self.console = Console()
        self.rating_module = RatingModule()
        
        # Configure optimizer
        if teleprompter is None:
            if optimizer == "random_search":
                self.teleprompter = BootstrapFewShotWithRandomSearch(
                    metric=self._validation_metric,
                    max_bootstrapped_demos=8,
                    max_labeled_demos=8,
                    num_candidate_programs=5
                )
            elif optimizer == "mipro":
                self.teleprompter = MIPROv2(
                    metric=self._validation_metric,
                    auto='light',
                )
            else:  # default bootstrap
                self.teleprompter = dspy.BootstrapFewShot(
                    metric=self._validation_metric,
                    max_bootstrapped_demos=8,
                    max_rounds=4,
                    max_labeled_demos=8
                )
        else:
            self.teleprompter = teleprompter
        
        # Load compiled model or initialize
        compiled_predictor = self._load_optimized_model()
        if not compiled_predictor:
            compiled_predictor = dspy.Predict(UnifiedTask)
            self.teleprompter.compile(
                compiled_predictor,
                trainset=self._load_training_data()
            )
            
        self.predictor = compiled_predictor
        
        # Parse the schema for validation
        self.output_schema_parser = etree.XMLSchema(etree.XML(OUTPUT_XML_SCHEMA))

    def _load_training_data(self):
        """Load training data from file."""
        try:
            with open("train_data.jsonl") as f:
                examples = []
                for line in f:
                    data = json.loads(line)
                    if "input_xml" not in data or "output_xml" not in data:
                        raise ValueError("Invalid training example format")
                    examples.append(dspy.Example(**data).with_inputs("input_xml"))
                return examples
        except FileNotFoundError:
            return [
                dspy.Example(
                    input_xml=EXAMPLE_INPUT_XML,
                    output_xml=EXAMPLE_OUTPUT_XML
                ).with_inputs("input_xml")
            ]

    def _load_optimized_model(self) -> dspy.Predict:
        """Load optimized model weights if available."""
        try:
            predictor = dspy.Predict(UnifiedTask)
            predictor.load("optimized_model.json")
            return predictor
        except FileNotFoundError:
            return None

    def save_optimized_model(self):
        """Save the optimized model weights."""
        self.predictor.save("optimized_model.json")
        self.console.print("Saved optimized model to optimized_model.json", style="bold green")

    def _validation_metric(self, example, pred, trace=None):
        """Custom metric that combines XML validity and quality ratings."""
        # First validate XML structure
        is_valid, validation_error = self.validate_xml(pred.output_xml)
        
        if not is_valid:
            self.console.print(f"[red]XML Validation Failed:[/red] {validation_error}", style="red")
            return 0.0  # Minimum score for invalid XML
        
        try:
            # Calculate quality rating if XML is valid
            raw_score = self.rating_module(
                pipeline_input=example.input_xml,
                pipeline_output=pred.output_xml
            )
            # Normalize 1-9 scale to 0-1 range
            normalized_score = raw_score / 9.0
            self.console.print(f"Quality Rating: {raw_score:.2f}/9 → {normalized_score:.2f}/1")
            return normalized_score
            
        except Exception as e:
            self.console.print(f"[yellow]Rating Error:[/yellow] {str(e)}", style="yellow")
            return 0.5  # Partial credit if rating fails

    def validate_xml(self, xml_string: str) -> tuple[bool, str]:
        """Validate XML against the schema.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Parse the XML
            xml_doc = etree.parse(io.StringIO(xml_string))
            
            # Validate against schema
            is_valid = self.output_schema_parser.validate(xml_doc)
            
            if not is_valid:
                error_message = self.output_schema_parser.error_log.filter_from_errors()[0]
                return False, str(error_message)
            
            return True, ""
        except Exception as e:
            return False, str(e)

    def forward(self, input_xml: str) -> str:
        """Generate the output XML based on the input XML."""
        result = self.predictor(input_xml=input_xml)
        output_xml = result.output_xml
        
        # Validate the output XML
        is_valid, error_message = self.validate_xml(output_xml)
        
        if not is_valid:
            # If invalid, try to fix common issues
            try:
                # Parse with more lenient parser
                root = ET.fromstring(output_xml)
                
                # Check for required elements
                required_elements = ["updated_memory", "new_plan", "execution_instructions", "is_done"]
                for elem_name in required_elements:
                    if root.find(elem_name) is None:
                        # Create missing element
                        if elem_name == "updated_memory":
                            ET.SubElement(root, elem_name).text = "No memory updates."
                        elif elem_name == "new_plan":
                            plan_elem = ET.SubElement(root, elem_name)
                            plan_root = ET.fromstring(EXAMPLE_PLAN_XML)
                            plan_elem.append(plan_root)
                        elif elem_name == "execution_instructions":
                            exec_elem = ET.SubElement(root, elem_name)
                            exec_root = ET.fromstring(EXAMPLE_EXECUTION_XML)
                            exec_elem.append(exec_root)
                        elif elem_name == "is_done":
                            ET.SubElement(root, elem_name).text = "false"
                
                # Convert back to string
                output_xml = ET.tostring(root, encoding='unicode')
                
                # Validate again
                is_valid, error_message = self.validate_xml(output_xml)
                if not is_valid:
                    self.console.print(f"Warning: Generated XML is still invalid: {error_message}", style="yellow")
            except Exception as e:
                self.console.print(f"Error fixing XML: {e}", style="red")
        
        return output_xml
