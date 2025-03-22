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
    input_schema = dspy.InputField(desc="Input XML schema", default=INPUT_XML_SCHEMA)
    output_schema = dspy.InputField(desc="Output XML schema", default=OUTPUT_XML_SCHEMA)
    input_xml = dspy.InputField(desc="Input XML following the input schema")
    output_xml = dspy.OutputField(desc="Output XML following the output schema")

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
                trainset=self._load_training_data(),
                requires_permission_to_run=False,
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
                    
                    # Handle both formats: with schemas in file or using defaults
                    if "input_schema" in data and "output_schema" in data:
                        example = dspy.Example(
                            input_schema=data["input_schema"],
                            output_schema=data["output_schema"],
                            input_xml=data["input_xml"],
                            output_xml=data["output_xml"]
                        )
                    else:
                        example = dspy.Example(
                            input_schema=INPUT_XML_SCHEMA,
                            output_schema=OUTPUT_XML_SCHEMA,
                            input_xml=data["input_xml"],
                            output_xml=data["output_xml"]
                        )
                    examples.append(example.with_inputs("input_schema", "output_schema", "input_xml"))
                return examples
        except FileNotFoundError:
            # Return empty list if file doesn't exist
            return []

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
        try:
            # First validate XML structure
            is_valid, validation_error = self.validate_xml(pred.output_xml)
            
            if not is_valid:
                self.console.print(f"[red]XML Validation Failed:[/red] {validation_error}", style="red")
                self.console.print(f"Invalid XML content:\n{pred.output_xml}", style="red")
                return 0.0
            
            # Calculate quality rating if XML is valid
            raw_score = self.rating_module(
                pipeline_input=example.input_xml,
                pipeline_output=pred.output_xml
            )
            normalized_score = raw_score / 9.0
            self.console.print(f"Quality Rating: {raw_score:.2f}/9 → {normalized_score:.2f}/1")
            return normalized_score
            
        except Exception as e:
            self.console.print(f"[bold red]Validation Error:[/bold red] {str(e)}", style="red")
            self.console.print(f"Example Input: {example.input_xml}", style="yellow")
            self.console.print(f"Predicted Output: {pred.output_xml}", style="yellow")
            return 0.0

    def validate_xml(self, xml_string: str) -> tuple[bool, str]:
        """Validate XML against the schema with local resolution"""
        try:
            # Use local schema instead of external URL
            schema_root = etree.XML(OUTPUT_XML_SCHEMA.encode())
            schema = etree.XMLSchema(schema_root)
            
            parser = etree.XMLParser(schema=schema)
            etree.fromstring(xml_string.encode(), parser)
            return True, ""
        except etree.XMLSyntaxError as e:
            return False, f"XML syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def forward(self, input_xml: str) -> str:
        """Generate the output XML based on the input XML."""
        self.console.print(f"Input XML:\n{input_xml}")
        result = self.predictor(
            input_schema=INPUT_XML_SCHEMA,
            output_schema=OUTPUT_XML_SCHEMA,
            input_xml=input_xml
        )
        output_xml = result.output_xml
        self.console.print(f"Generated output XML:\n{output_xml}")
        
        # Validate the output XML
        is_valid, error_message = self.validate_xml(output_xml)
        
        if not is_valid:
            # If invalid, try to fix common issues
            try:
                # Parse with more lenient parser
                root = ET.fromstring(output_xml)
                
                
                # Convert back to string
                output_xml = ET.tostring(root, encoding='unicode')
                
                # Validate again
                is_valid, error_message = self.validate_xml(output_xml)
                if not is_valid:
                    self.console.print(f"Warning: Generated XML is still invalid: {error_message}", style="yellow")
            except Exception as e:
                self.console.print(f"Error fixing XML: {e}", style="red")
        
        return output_xml
