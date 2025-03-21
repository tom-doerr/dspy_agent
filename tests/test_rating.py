import pytest
import dspy
from dspy_agent.rating import RatingModule

@pytest.fixture
def setup_model():
    # Use a mock LM for testing
    lm = dspy.MockLM({
        "Rate this response": "7"
    })
    dspy.settings.configure(lm=lm)
    yield
    # Reset settings after test
    dspy.settings.configure(lm=None)

def test_rating_module(setup_model):
    rating_module = RatingModule()
    rating = rating_module(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Accuracy"
    )
    assert 1 <= rating <= 9
