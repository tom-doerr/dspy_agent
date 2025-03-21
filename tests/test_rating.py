import pytest
import dspy
from dspy_agent.rating import RatingModule

class TestRating:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        # Create a simple predictable LM for testing
        class PredictableLM(dspy.LM):
            def __call__(self, prompt, **kwargs):
                return {"response": "7"}
        
        self.lm = PredictableLM()
        dspy.settings.configure(lm=self.lm)
        yield
        # Reset settings after test
        dspy.settings.configure(lm=None)
    
    def test_rating_module(self):
        rating_module = RatingModule()
        rating = rating_module(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "Accuracy"
        )
        assert 1 <= rating <= 9
