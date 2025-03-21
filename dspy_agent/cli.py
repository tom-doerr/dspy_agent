import typer
from rich.console import Console
import xml.etree.ElementTree as ET
from .unified import UnifiedModule

app = typer.Typer()
console = Console()

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

if __name__ == "__main__":
    app()
