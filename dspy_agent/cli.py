import typer
from rich.console import Console
from rich.panel import Panel
from typing import Optional
from .pipeline import SimplePipeline
from .rating import RatingModule
from .optimizer import PipelineOptimizer

app = typer.Typer()
console = Console()

@app.command()
def run(
    task: str = typer.Argument(..., help="The task to perform"),
    criterion: Optional[str] = typer.Option(None, help="The criterion for rating the output"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use")
):
    """Run the DSPy agent on a specific task."""
    import dspy
    
    # Initialize the model
    lm = dspy.LM(model)
    dspy.settings.configure(lm=lm)
    
    # Initialize the pipeline
    pipeline = SimplePipeline()
    
    # Process the task
    result = pipeline(task)
    
    # Display the result
    console.print(Panel(result, title="Result", border_style="green"))
    
    # Rate the result if criterion is provided
    if criterion:
        rating_module = RatingModule()
        rating = rating_module(task, result, criterion)
        console.print(f"Rating ({criterion}): {rating}/9", style="bold blue")

@app.command()
def optimize(
    task: str = typer.Argument(..., help="The task to optimize for"),
    criterion: str = typer.Argument(..., help="The criterion for optimization"),
    iterations: int = typer.Option(3, help="Number of optimization iterations"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use")
):
    """Optimize the pipeline for a specific task and criterion."""
    # Initialize the optimizer
    optimizer = PipelineOptimizer(model_name=model)
    
    # Run the optimization
    console.print(f"Optimizing pipeline for task: {task}", style="bold")
    console.print(f"Criterion: {criterion}", style="italic")
    
    optimized_pipeline = optimizer.optimize(task, criterion, num_iterations=iterations)
    
    # Test the optimized pipeline
    result = optimized_pipeline(task)
    
    # Display the result
    console.print(Panel(result, title="Optimized Result", border_style="green"))
    
    # Rate the result
    rating = optimizer.rating_module(task, result, criterion)
    console.print(f"Rating ({criterion}): {rating}/9", style="bold blue")

if __name__ == "__main__":
    app()
