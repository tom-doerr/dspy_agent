import dspy
from typing import Dict, Any, Optional, List
from .memory import MemoryModule

class SimpleTask(dspy.Signature):
    """A simple task signature with input and output."""
    
    input = dspy.InputField()
    context = dspy.InputField(desc="Optional context information")
    output = dspy.OutputField()

class SimplePipeline(dspy.Module):
    """A simple DSPy pipeline that processes input and generates output."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleTask)
        self.memory_module = MemoryModule()
    
    def forward(self, input_text: str, context: Optional[Dict[str, Any]] = None, stream: bool = False):
        """Process the input and return the output."""
        context = context or {}
        context_str = str(context)
        
        # Update memory with the input
        memory_ops = self.memory_module(f"New task: {input_text}")
        self.memory_module.update_memory(memory_ops)
        
        # Include memory in context
        context_with_memory = f"Memory: {self.memory_module.memory}\nContext: {context_str}"
        
        # Get prediction
        result = self.predictor(input=input_text, context=context_with_memory)
        
        # Update memory with the result
        memory_ops = self.memory_module(f"Task result: {result.output}")
        self.memory_module.update_memory(memory_ops)
        
        if stream:
            # Simulate streaming by yielding words
            for word in result.output.split():
                yield word + " "
        else:
            return result.output
    
    def get_memory(self) -> str:
        """Get the current memory state."""
        return self.memory_module.memory
