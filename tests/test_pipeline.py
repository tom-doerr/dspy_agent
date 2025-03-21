import pytest
import dspy
from dspy_agent.pipeline import SimplePipeline

class TestPipeline:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        # Create a simple predictable LM for testing
        class PredictableLM(dspy.LM):
            def __init__(self):
                super().__init__(model="mock")
                
            def __call__(self, prompt, **kwargs):
                return {"response": "The capital of France is Paris."}
        
        self.lm = PredictableLM()
        dspy.settings.configure(lm=self.lm)
        yield
        # Reset settings after test
        dspy.settings.configure(lm=None)
    
    def test_simple_pipeline(self):
        pipeline = SimplePipeline()
        result = pipeline("What is the capital of France?")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_streaming(self):
        pipeline = SimplePipeline()
        tokens = list(pipeline("What is the capital of France?", stream=True))
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
