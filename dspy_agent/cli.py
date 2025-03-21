import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from typing import Optional, Callable
import xml.etree.ElementTree as ET
from .pipeline import SimplePipeline
from .rating import RatingModule
from .optimizer import PipelineOptimizer
from .unified import UnifiedModule

app = typer.Typer()
console = Console()

@app.command()
def run_pipeline(
    task: str = typer.Argument(..., help="The task to perform"),
    criterion: Optional[str] = typer.Option(None, help="The criterion for rating the output"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use"),
    stream: bool = typer.Option(False, "--stream", help="Stream output in real-time")
):
    """Run the DSPy agent on a specific task using the SimplePipeline."""
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
    console.print(f"Memory before: {pipeline.get_memory()}", style="blue")
    
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
    
    console.print(f"Memory after: {pipeline.get_memory()}", style="blue")
    
    # Rate the result if criterion is provided
    if criterion:
        rating_module = RatingModule()
        rating = rating_module(task, result, criterion)
        console.print(f"Rating ({criterion}): {rating}/9", style="bold blue")

@app.command()
def run(
    task: str = typer.Argument(..., help="The task to perform"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use"),
    loop: int = typer.Option(1, "--loop", help="Number of loop iterations"),
):
    """Run the DSPy agent with a unified module for memory, planning, and execution."""
    import dspy
    
    if not task.strip():
        console.print("Error: Task cannot be empty", style="bold red")
        raise typer.Exit(code=1)

    # Configure DSPy with the language model
    lm = dspy.LM(model)
    dspy.settings.configure(lm=lm)
    unified_module = UnifiedModule()

    # Initial state
    memory = ""
    last_plan = "<plan></plan>"
    last_action = ""
    observation = task  # Start with the task as the initial observation

    for i in range(loop):
        console.print(f"\nLoop iteration {i+1}/{loop}", style="bold")

        # Construct the input XML
        input_xml = f"""
        <agent_state>
            <memory>{memory}</memory>
            <last_plan>{last_plan}</last_plan>
            <last_action>{last_action}</last_action>
            <observation>{observation}</observation>
        </agent_state>
        """

        # Get the output XML from the unified module
        output_xml = unified_module(input_xml)

        # Parse the output XML
        try:
            root = ET.fromstring(output_xml)
            updated_memory = root.find("updated_memory").text or ""
            new_plan = root.find("new_plan").text or ""
            execution_instructions = root.find("execution_instructions").text or ""
        except Exception as e:
            console.print(f"Error parsing output XML: {e}", style="bold red")
            console.print(f"Output received: {output_xml}", style="red")
            break

        # Simulate executing the instructions and getting a new observation
        # TODO: Replace with real execution (e.g., run commands, collect output)
        observation = f"Processed observation from iteration {i+1}"

        # Update the state for the next iteration
        memory = updated_memory
        last_plan = new_plan
        last_action = "executed_instructions"  # Could parse from execution_instructions

        # Display the results
        console.print(f"Input XML: {input_xml}", style="dim")
        console.print(f"Output XML: {output_xml}", style="dim")
        console.print(f"Memory: {memory}", style="blue")
        console.print(f"Plan: {new_plan}", style="green")
        console.print(f"Execution Instructions: {execution_instructions}", style="yellow")

    console.print("\nAgent run completed", style="bold green")

@app.command()
def memory(
    action: str = typer.Argument(..., help="Action to perform: 'show', 'add', or 'clear'"),
    content: str = typer.Option(None, help="Content to add to memory (for 'add' action)"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use")
):
    """View or manipulate the agent's memory."""
    import dspy
    
    # Initialize the model
    lm = dspy.LM(model)
    dspy.settings.configure(lm=lm)
    
    # Initialize the pipeline
    pipeline = SimplePipeline()
    
    if action.lower() == "show":
        memory = pipeline.get_memory()
        if not memory:
            console.print("Memory is empty", style="yellow")
        else:
            console.print(Panel(memory, title="Current Memory", border_style="blue"))
    
    elif action.lower() == "add" and content:
        memory_before = pipeline.get_memory()
        memory_ops = pipeline.memory_module(content)
        updated_memory = pipeline.memory_module.update_memory(memory_ops)
        
        console.print(f"Memory before: {memory_before}", style="blue")
        console.print(f"Added: {content}", style="green")
        console.print(f"Memory after: {updated_memory}", style="blue")
        
        # Show the operations that were performed
        console.print("Operations performed:", style="yellow")
        for op in memory_ops:
            console.print(f"  Search: '{op.get('search', '')}' → Replace: '{op.get('replace', '')}'")
    
    elif action.lower() == "clear":
        pipeline.memory_module.memory = ""
        console.print("Memory cleared", style="green")
    
    else:
        console.print(f"Unknown action: {action}", style="bold red")
        console.print("Available actions: show, add, clear", style="yellow")

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
