import dspy


class RatingTask(dspy.Signature):
    """Rate pipeline output across multiple criteria.
    Be harsh in your rating, deduct a point for every issue.
    Provide detailed reasoning for each score."""

    pipeline_input = dspy.InputField()
    pipeline_output = dspy.InputField()

    memory_reasoning = dspy.OutputField(desc="Reasoning for memory score")
    added_all_relevant_information_to_memory_score = dspy.OutputField(
        desc="Did the pipeline add all relevant information to memory? (1-9)"
    )

    action_reasoning = dspy.OutputField(desc="Reasoning for next action score")
    next_action_score = dspy.OutputField(desc="How good is the next action? (1-9)")

    plan_reasoning = dspy.OutputField(desc="Reasoning for plan score")
    plan_score = dspy.OutputField(desc="How good is the plan? (1-9)")


class RatingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rater = dspy.Predict(RatingTask)

    def forward(self, pipeline_input: str, pipeline_output: str) -> float:
        """Rate the output and return the average score."""
        result = self.rater(
            pipeline_input=pipeline_input, pipeline_output=pipeline_output
        )
        try:
            scores = [
                int(result.added_all_relevant_information_to_memory_score),
                int(result.next_action_score),
                int(result.plan_score),
            ]
            print(f"Memory: {scores[0]}, Action: {scores[1]}, Plan: {scores[2]}")
            # Clamp scores between 1 and 9
            scores = [max(1, min(9, score)) for score in scores]
            return sum(scores) / len(scores)  # Average
        except (ValueError, AttributeError):
            return 5.0  # Default if parsing fails

    def get_detailed_ratings(self, pipeline_input: str, pipeline_output: str) -> dict:
        """Get detailed ratings with reasoning."""
        result = self.rater(
            pipeline_input=pipeline_input, pipeline_output=pipeline_output
        )
        try:
            return {
                "memory": {
                    "score": int(
                        max(
                            1,
                            min(
                                9,
                                int(
                                    result.added_all_relevant_information_to_memory_score
                                ),
                            ),
                        )
                    ),
                    "reasoning": result.memory_reasoning,
                },
                "action": {
                    "score": int(max(1, min(9, int(result.next_action_score)))),
                    "reasoning": result.action_reasoning,
                },
                "plan": {
                    "score": int(max(1, min(9, int(result.plan_score)))),
                    "reasoning": result.plan_reasoning,
                },
            }
        except (ValueError, AttributeError) as e:
            return {
                "error": str(e),
                "memory": {"score": 5, "reasoning": "Error parsing rating"},
                "action": {"score": 5, "reasoning": "Error parsing rating"},
                "plan": {"score": 5, "reasoning": "Error parsing rating"},
            }
