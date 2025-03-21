import dspy
from .schema import (
    INPUT_XML_SCHEMA,
    OUTPUT_XML_SCHEMA,
    EXAMPLE_INPUT_XML,
    EXAMPLE_OUTPUT_XML
)

class UnifiedTask(dspy.Signature):
    """Generate output XML with updated memory, new plan, and execution instructions from input XML."""
    input_xml = dspy.InputField(desc=f"Input XML with memory, last_plan, last_action, observation. Schema: {INPUT_XML_SCHEMA}")
    output_xml = dspy.OutputField(desc=f"Output XML with updated_memory, new_plan, execution_instructions. Schema: {OUTPUT_XML_SCHEMA}")

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

    def forward(self, input_xml: str) -> str:
        """Generate the output XML based on the input XML."""
        result = self.predictor(input_xml=input_xml)
        return result.output_xml
