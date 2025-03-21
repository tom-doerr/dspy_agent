import pytest
import dspy
from dspy_agent.pipeline import SimplePipeline

@pytest.fixture
def setup_model():
    # Use a mock LM for testing
    lm = dspy.MockLM({
        "What is the capital of France?": "The capital of France is Paris."
    })
    dspy.settings.configure(lm=lm)
    yield
    # Reset settings after test
    dspy.settings.configure(lm=None)

def test_simple_pipeline(setup_model):
    pipeline = SimplePipeline()
    result = pipeline("What is the capital of France?")
    assert "Paris" in result
