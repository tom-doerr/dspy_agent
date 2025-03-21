import dspy
from typing import List, Dict, Any

class MemoryTask(dspy.Signature):
    """Generate search-and-replace operations to update memory based on an observation."""
    memory = dspy.InputField(desc="Current memory as a string")
    observation = dspy.InputField(desc="New information to incorporate")
    search_replace_ops = dspy.OutputField(desc="List of JSON dicts with 'search' and 'replace' keys")

class MemoryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(MemoryTask)
        self.memory = ""  # Single string memory

    def forward(self, observation: str) -> List[dict]:
        """Process the observation and return search-replace operations."""
        result = self.predictor(memory=self.memory, observation=observation)
        try:
            ops = dspy.from_json(result.search_replace_ops)
            return ops if isinstance(ops, list) else [ops]
        except Exception as e:
            # Fallback if JSON parsing fails
            return [{"search": "", "replace": observation}]

    def update_memory(self, ops: List[dict]):
        """Apply search-replace ops to memory."""
        updated_memory = self.memory
        for op in ops:
            if "search" in op and "replace" in op:
                updated_memory = updated_memory.replace(op["search"], op["replace"])
        self.memory = updated_memory
        return self.memory
