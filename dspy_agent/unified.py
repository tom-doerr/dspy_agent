import dspy
import json
from rich.console import Console
from .rating import RatingModule
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
import xml.etree.ElementTree as ET
from lxml import etree
import io
import time
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
    """Generate output XML with updated memory, new plan, and execution instructions from input XML. """
    input_schema = dspy.InputField(desc="Input XML schema", default=INPUT_XML_SCHEMA)
    output_schema = dspy.InputField(desc="Output XML schema", default=OUTPUT_XML_SCHEMA)
    input_xml = dspy.InputField(desc="Input XML following the input schema")
    output_xml = dspy.OutputField(desc="Output XML following the output schema")

class UnifiedModule(dspy.Module):
    def __init__(self, predictor=None):
        super().__init__()
        self.console = Console()
        self.rating_module = RatingModule()
        self.predictor = predictor or dspy.Predict(UnifiedTask)
        
        # Parse the schema for validation
        self.output_schema_parser = etree.XMLSchema(etree.XML(OUTPUT_XML_SCHEMA))



    def _validation_metric(self, example, pred, trace=None):
        """Custom metric that combines XML validity and quality ratings."""
        self.console.print(f'Generated XML: {pred.output_xml}', flush=True)
        print(f'Generated XML: {pred.output_xml}', flush=True)
        # time.sleep(1)

        # First validate XML structure
        is_valid, validation_error = self.validate_xml(pred.output_xml)
        
        if not is_valid:
            self.console.print(f"[red]XML Validation Failed:[/red] {validation_error}", style="red")
            self.console.print(f"Invalid XML content:\n{pred.output_xml}", style="red")
            return 0.0
        
        # Get detailed ratings with reasoning
        detailed_ratings = self.rating_module.get_detailed_ratings(
            pipeline_input=example.input_xml,
            pipeline_output=pred.output_xml
        )
        
        # Print detailed ratings
        self.console.print("[bold]Detailed Ratings:[/bold]")
        for criterion, rating in detailed_ratings.items():
            if criterion != "error":
                self.console.print(f"[bold]{criterion.capitalize()}:[/bold] {rating['score']}/9")
                self.console.print(f"  Reasoning: {rating['reasoning']}")
        
        # Calculate quality rating if XML is valid
        raw_score = self.rating_module(
            pipeline_input=example.input_xml,
            pipeline_output=pred.output_xml
        )
        normalized_score = raw_score / 9.0
        self.console.print(f"Quality Rating: {raw_score:.2f}/9 → {normalized_score:.2f}/1")
        return normalized_score

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
        self.console.print(f"Input XML:\n{input_xml}", flush=True)
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
