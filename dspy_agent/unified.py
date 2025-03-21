import dspy
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
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(UnifiedTask)
        
        # Add examples to help the model understand the expected format
        self.predictor.config.examples = [
            dspy.Example(
                input_xml=EXAMPLE_INPUT_XML,
                output_xml=EXAMPLE_OUTPUT_XML
            )
        ]
        
        # Parse the schema for validation
        self.output_schema_parser = etree.XMLSchema(etree.XML(OUTPUT_XML_SCHEMA))

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
                    print(f"Warning: Generated XML is still invalid: {error_message}")
            except Exception as e:
                print(f"Error fixing XML: {e}")
        
        return output_xml
