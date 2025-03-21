import dspy

class RatingTask(dspy.Signature):
    """Rate pipeline output across multiple criteria."""
    pipeline_input = dspy.InputField()
    pipeline_output = dspy.InputField()
    accuracy = dspy.OutputField(desc="Score for accuracy (1-9)")
    clarity = dspy.OutputField(desc="Score for clarity (1-9)")
    relevance = dspy.OutputField(desc="Score for relevance (1-9)")

class RatingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rater = dspy.Predict(RatingTask)

    def forward(self, pipeline_input: str, pipeline_output: str) -> float:
        """Rate the output and return the average score."""
        result = self.rater(
            pipeline_input=pipeline_input,
            pipeline_output=pipeline_output
        )
        try:
            scores = [
                int(result.accuracy),
                int(result.clarity),
                int(result.relevance)
            ]
            # Clamp scores between 1 and 9
            scores = [max(1, min(9, score)) for score in scores]
            return sum(scores) / len(scores)  # Average
        except (ValueError, AttributeError):
            return 5.0  # Default if parsing fails
