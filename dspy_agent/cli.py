import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from typing import Optional, Callable
from .pipeline import SimplePipeline
from .rating import RatingModule
from .optimizer import PipelineOptimizer

app = typer.Typer()
console = Console()

@app.command()
def run(
    task: str = typer.Argument(..., help="The task to perform"),
    criterion: Optional[str] = typer.Option(None, help="The criterion for rating the output"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use"),
    stream: bool = typer.Option(False, "--stream", help="Stream output in real-time")
):
    """Run the DSPy agent on a specific task."""
    import dspy
    
    # Validate inputs
    if not task.strip():
        console.print("Error: Task cannot be empty", style="bold red")
        raise typer.Exit(code=1)
    
    if criterion and not criterion.strip():
        console.print("Error: Criterion cannot be empty if provided", style="bold red")
        raise typer.Exit(code=1)
    
    # Initialize the model
    lm = dspy.LM(model)
    dspy.settings.configure(lm=lm)
    
    # Initialize the pipeline
    pipeline = SimplePipeline()
    
    # Process the task with or without streaming
    if stream:
        console.print("Streaming output:", style="bold")
        result_text = ""
        for token in pipeline(task, stream=True):
            result_text += token
            console.print(token, end="", style="green")
        console.print()  # Add a newline at the end
        result = result_text
    else:
        result = pipeline(task)
        # Handle if result is a generator
        if hasattr(result, '__iter__') and not isinstance(result, str):
            result = "".join(list(result))
        console.print(Panel(str(result), title="Result", border_style="green"))
    
    # Rate the result if criterion is provided
    if criterion:
        rating_module = RatingModule()
        rating = rating_module(task, result, criterion)
        console.print(f"Rating ({criterion}): {rating}/9", style="bold blue")

@app.command()
def optimize(
    task: str = typer.Argument(..., help="The task to optimize for"),
    criterion: str = typer.Option(..., help="The criterion for optimization"),
    iterations: int = typer.Option(3, help="Number of optimization iterations"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use")
):
    """Optimize the pipeline for a specific task and criterion."""
    # Validate inputs
    if not task.strip() or not criterion.strip():
        console.print("Error: Task and criterion must not be empty", style="bold red")
        raise typer.Exit(code=1)
    
    # Initialize the optimizer
    optimizer = PipelineOptimizer(model_name=model)
    
    # Run the optimization with progress bar
    console.print(f"Optimizing pipeline for task: {task}", style="bold")
    console.print(f"Criterion: {criterion}", style="italic")
    
    with Progress() as progress:
        task_progress = progress.add_task("Optimizing", total=iterations)
        
        def update_progress():
            progress.update(task_progress, advance=1)
        
        optimized_pipeline = optimizer.optimize(
            task, 
            criterion, 
            num_iterations=iterations,
            progress_callback=update_progress
        )
    
    # Test the optimized pipeline
    result = optimized_pipeline(task)
    
    # Display the result
    console.print(Panel(result, title="Optimized Result", border_style="green"))
    
    # Rate the result
    rating = optimizer.rating_module(task, result, criterion)
    console.print(f"Rating ({criterion}): {rating}/9", style="bold blue")

if __name__ == "__main__":
    app()
