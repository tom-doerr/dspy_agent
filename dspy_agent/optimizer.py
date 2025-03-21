import dspy
from typing import List, Tuple, Dict, Any
from .pipeline import SimplePipeline
from .rating import RatingModule

class PipelineOptimizer:
    """Optimizes the pipeline using bootstrap few-shot examples."""
    
    def __init__(self, model_name: str = "deepseek/deepseek-chat"):
        self.lm = dspy.LM(model_name)
        dspy.settings.configure(lm=self.lm)
        self.pipeline = SimplePipeline()
        self.rating_module = RatingModule()
    
    def generate_examples(self, task: str, criterion: str, num_examples: int = 5) -> List[Dict[str, Any]]:
        """Generate examples for bootstrapping."""
        examples = []
        
        # Create a task prompt that will generate diverse inputs
        task_prompt = f"Generate {num_examples} diverse examples for the task: {task}"
        inputs = self.lm.generate(task_prompt, max_tokens=1000).split("\n\n")
        
        for i, input_text in enumerate(inputs[:num_examples]):
            if not input_text.strip():
                continue
                
            # Get output from the current pipeline
            output = self.pipeline(input_text)
            
            # Get rating for this output
            rating = self.rating_module(input_text, output, criterion)
            
            examples.append({
                "input": input_text,
                "output": output,
                "rating": rating
            })
            
        return examples
    
    def optimize(self, task: str, criterion: str, num_iterations: int = 3) -> SimplePipeline:
        """Optimize the pipeline using bootstrap few-shot learning."""
        
        for iteration in range(num_iterations):
            print(f"Optimization iteration {iteration+1}/{num_iterations}")
            
            # Generate examples
            examples = self.generate_examples(task, criterion)
            
            # Sort examples by rating
            examples.sort(key=lambda x: x["rating"], reverse=True)
            
            # Use top examples for bootstrapping
            top_examples = examples[:3]
            
            # Create a bootstrapped optimizer
            teleprompter = dspy.Teleprompter(dspy.ChainOfThought(SimplePipeline))
            
            # Prepare training data
            train_data = []
            for ex in top_examples:
                train_data.append({"input": ex["input"], "output": ex["output"]})
            
            # Optimize with the examples
            optimized_pipeline = teleprompter.compile(
                SimplePipeline(),
                trainset=train_data,
                metric=lambda example, pred: self.rating_module(
                    example["input"], 
                    pred.output, 
                    criterion
                ) / 9.0  # Normalize to 0-1 range
            )
            
            # Update the pipeline
            self.pipeline = optimized_pipeline
            
        return self.pipeline
