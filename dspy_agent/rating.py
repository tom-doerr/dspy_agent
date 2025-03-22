import dspy
from rich import print

class RatingTask(dspy.Signature):
    """Rate pipeline output across multiple criteria.
    Be hash in your rating, deduct a point for every issue."""
    pipeline_input = dspy.InputField()
    pipeline_output = dspy.InputField()
    added_all_relevant_information_to_memory_score = dspy.OutputField(desc="Did the pipeline add all relevant information to memory?")
    next_action_score = dspy.OutputField(desc="How good is the next action?")
    plan_score = dspy.OutputField(desc="How good is the plan?")


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
                result.added_all_relevant_information_to_memory_score,
                result.next_action_score,
                result.plan_score
            ]
            print(f"Accuracy: {scores[0]}, Clarity: {scores[1]}, Relevance: {scores[2]}")
            # Clamp scores between 1 and 9
            scores = [max(1, min(9, score)) for score in scores]
            return sum(scores) / len(scores)  # Average
        except (ValueError, AttributeError):
            return 5.0  # Default if parsing fails
