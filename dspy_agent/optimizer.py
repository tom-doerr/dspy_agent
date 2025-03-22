import dspy
import json
import os
from rich.console import Console
from .unified import UnifiedModule
from .rating import RatingModule
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2

console = Console()

class Optimizer:
    """Handles model optimization workflow"""
    
    def __init__(self, model_name: str = "deepseek/deepseek-chat"):
        self.model_name = model_name
        self.rating_module = RatingModule()
        self._configure_model()
        
    def _configure_model(self):
        """Centralized model configuration"""
        lm = dspy.LM(self.model_name)
        dspy.settings.configure(lm=lm)
    
    def _load_training_data(self, data_path: str):
        """Load training data from file"""
        if not os.path.exists(data_path):
            console.print(f"Error: Training data file '{data_path}' not found", style="bold red")
            raise FileNotFoundError(f"Training data not found at {data_path}")
            
        with open(data_path) as f:
            return [self._parse_training_example(line) for line in f]
    
    def _parse_training_example(self, line: str):
        """Parse a single training example"""
        data = json.loads(line)
        return dspy.Example(
            input_schema=data.get("input_schema", INPUT_XML_SCHEMA),
            output_schema=data.get("output_schema", OUTPUT_XML_SCHEMA),
            input_xml=data["input_xml"],
            output_xml=data.get("output_xml", "")
        ).with_inputs("input_schema", "output_schema", "input_xml")
    
    def optimize(self, training_data_path: str, optimizer_type: str = "bootstrap", num_iterations: int = 3):
        """Run full optimization workflow"""
        train_data = self._load_training_data(training_data_path)
        module = UnifiedModule(optimizer=optimizer_type)
        
        try:
            optimized_predictor = module.teleprompter.compile(
                module.predictor,
                trainset=train_data,
                requires_permission_to_run=False,
            )
            module.predictor = optimized_predictor
            module.save_optimized_model()
            console.print(f"Optimization complete with {len(train_data)} examples", style="bold green")
            return module
        except Exception as e:
            console.print(f"[bold red]Optimization Failed:[/bold red] {str(e)}", style="red")
            raise
