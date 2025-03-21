import pytest
import dspy
from dspy_agent.memory import MemoryModule

class TestMemory:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        # Create a simple predictable LM for testing
        class PredictableLM(dspy.LM):
            def __init__(self):
                super().__init__(model="mock")
                
            def __call__(self, prompt, **kwargs):
                # Return a simple search-replace operation
                return {"response": '[{"search": "old_value", "replace": "new_value"}]'}
        
        self.lm = PredictableLM()
        dspy.settings.configure(lm=self.lm)
        yield
        # Reset settings after test
        dspy.settings.configure(lm=None)
    
    def test_memory_module_creation(self):
        memory = MemoryModule()
        assert memory.memory == ""
    
    def test_memory_update(self):
        memory = MemoryModule()
        memory.memory = "This contains old_value in it."
        
        # Generate operations
        ops = memory("New observation")
        
        # Apply operations
        updated = memory.update_memory(ops)
        
        assert "new_value" in updated
        assert memory.memory == updated
