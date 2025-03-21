import dspy

class RatingTask(dspy.Signature):
    """A signature for rating pipeline outputs."""
    
    pipeline_input = dspy.InputField()
    pipeline_output = dspy.InputField()
    criterion = dspy.InputField()
    rating = dspy.OutputField(desc="A rating from 1 to 9 where 1 is worst and 9 is best")

class RatingModule(dspy.Module):
    """Module that rates pipeline outputs based on specified criteria."""
    
    def __init__(self):
        super().__init__()
        self.rater = dspy.Predict(RatingTask)
    
    def forward(self, pipeline_input: str, pipeline_output: str, criterion: str) -> int:
        """Rate the pipeline output based on the criterion."""
        result = self.rater(
            pipeline_input=pipeline_input,
            pipeline_output=pipeline_output,
            criterion=criterion
        )
        # Convert the rating to an integer between 1 and 9
        try:
            rating = int(result.rating)
            return max(1, min(9, rating))
        except ValueError:
            # Default to middle rating if conversion fails
            return 5
