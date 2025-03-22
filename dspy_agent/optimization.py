import dspy
import json
import os
from rich.console import Console
from .unified import UnifiedModule, UnifiedTask
from .rating import RatingModule
from .schema import INPUT_XML_SCHEMA, OUTPUT_XML_SCHEMA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2

console = Console()

class Optimizer:
    """Handles model optimization workflow"""
    
    def __init__(self, model_name: str = "deepseek/deepseek-chat", optimizer_type: str = "bootstrap"):
        self.model_name = model_name
        self.rating_module = RatingModule()
        self.optimizer = None
        self.optimizer_type = optimizer_type
        self._configure_model()
        self._init_teleprompter(optimizer_type)

    def _init_teleprompter(self, optimizer_type="bootstrap"):
        """Initialize the optimization strategy"""
        if optimizer_type == "random_search":
            self.optimizer = BootstrapFewShotWithRandomSearch(
                metric=self._validation_metric,
                max_bootstrapped_demos=8,
                max_labeled_demos=8,
                num_candidate_programs=5
            )
        elif optimizer_type == "mipro":
            self.optimizer = MIPROv2(
                metric=self._validation_metric,
                auto='light',
            )
        else:  # default bootstrap
            self.optimizer = dspy.BootstrapFewShot(
                metric=self._validation_metric,
                max_bootstrapped_demos=8,
                max_rounds=4,
                max_labeled_demos=8
            )

    def _validation_metric(self, example, pred, trace=None):
        """Custom metric that combines XML validity and quality ratings."""
        # Validate XML structure
        is_valid, error = UnifiedModule().validate_xml(pred.output_xml)
        if not is_valid:
            console.print(f"[red]XML Validation Failed:[/red] {error}", style="red")
            return 0.0
        
        # Get detailed ratings with reasoning
        detailed_ratings = self.rating_module.get_detailed_ratings(
            pipeline_input=example.input_xml,
            pipeline_output=pred.output_xml
        )
        
        # Print detailed ratings
        console.print("[bold]Detailed Ratings:[/bold]")
        for criterion, rating in detailed_ratings.items():
            if criterion != "error":
                console.print(f"[bold]{criterion.capitalize()}:[/bold] {rating['score']}/9")
                console.print(f"  Reasoning: {rating['reasoning']}")
        
        # Calculate quality rating
        raw_score = self.rating_module(
            pipeline_input=example.input_xml,
            pipeline_output=pred.output_xml
        )
        return raw_score / 9.0

    def _load_optimized_model(self) -> dspy.Predict:
        """Load optimized model weights if available."""
        try:
            predictor = dspy.Predict(UnifiedTask)
            predictor.load("optimized_model.json")
            return predictor
        except FileNotFoundError:
            return None

    def save_optimized_model(self, predictor):
        """Save the optimized model weights."""
        predictor.save("optimized_model.json")
        
    def _configure_model(self):
        """Centralized model configuration"""
        from .config import configure_lm
        configure_lm(self.model_name)
    
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
    
    def optimize(self, training_data_path: str, num_iterations: int = 3):
        """Run full optimization workflow"""
        train_data = self._load_training_data(training_data_path)
        
        try:
            # Start with base predictor or pre-optimized version
            predictor = self._load_optimized_model() or dspy.Predict(UnifiedTask)
            
            # Run optimization
            # Only MIPROv2 needs permission to run
            requires_permission = self.optimizer_type == "mipro"
            optimized_predictor = self.optimizer.compile(
                predictor,
                trainset=train_data,
                requires_permission_to_run=False,
            )
            
            # Save and return optimized model
            self.save_optimized_model(optimized_predictor)
            console.print(f"Optimization complete with {len(train_data)} examples", style="bold green")
            return UnifiedModule(optimized_predictor)
        except Exception as e:
            console.print(f"[bold red]Optimization Failed:[/bold red] {str(e)}", style="red")
            raise
