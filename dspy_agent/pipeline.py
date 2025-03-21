import dspy

class SimpleTask(dspy.Signature):
    """A simple task signature with input and output."""
    
    input = dspy.InputField()
    output = dspy.OutputField()

class SimplePipeline(dspy.Module):
    """A simple DSPy pipeline that processes input and generates output."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleTask)
    
    def forward(self, input_text: str, stream: bool = False):
        """Process the input and return the output."""
        result = self.predictor(input=input_text)
        
        if stream:
            # Simulate streaming by yielding words
            for word in result.output.split():
                yield word + " "
        else:
            return result.output
