import dspy

class UnifiedTask(dspy.Signature):
    """Generate output XML with updated memory, new plan, and execution instructions from input XML."""
    input_xml = dspy.InputField(desc="Input XML with memory, last_plan, last_action, observation")
    output_xml = dspy.OutputField(desc="Output XML with updated_memory, new_plan, execution_instructions")

class UnifiedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(UnifiedTask)

    def forward(self, input_xml: str) -> str:
        """Generate the output XML based on the input XML."""
        result = self.predictor(input_xml=input_xml)
        return result.output_xml
