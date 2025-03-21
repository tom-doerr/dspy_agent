import typer
from rich.console import Console
import xml.etree.ElementTree as ET
from .unified import UnifiedModule
from .schema import INPUT_XML_SCHEMA, OUTPUT_XML_SCHEMA

app = typer.Typer()
console = Console()

@app.command()
def run(
    task: str = typer.Argument(..., help="The task to perform"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use"),
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
    
    iteration = 1
    is_done = False

    while not is_done:
        console.print(f"\nLoop iteration {iteration}", style="bold")
        iteration += 1

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
            
            # Get the plan as XML string
            new_plan_elem = root.find("new_plan")
            new_plan = ET.tostring(new_plan_elem, encoding='unicode') if new_plan_elem is not None else "<plan></plan>"
            
            # Get the execution instructions
            exec_instructions_elem = root.find("execution_instructions")
            execution_instructions = ET.tostring(exec_instructions_elem, encoding='unicode') if exec_instructions_elem is not None else ""
            
            # Check if task is done
            is_done_element = root.find("is_done")
            is_done = is_done_element is not None and is_done_element.text.lower() == "true"
            
            # Extract operations for potential execution
            operations = []
            write_ops = root.find(".//write_operations")
            if write_ops is not None:
                for op in write_ops.findall("operation"):
                    op_type = op.get("type")
                    if op_type == "command":
                        cmd = op.get("command") or op.text
                        operations.append(("command", cmd))
                    elif op_type == "file":
                        path = op.get("path")
                        content = op.text or ""
                        operations.append(("file", path, content))
                    elif op_type == "message":
                        operations.append(("message", op.text or ""))
            
        except Exception as e:
            console.print(f"Error parsing output XML: {e}", style="bold red")
            console.print(f"Output received: {output_xml}", style="red")
            break

        # Simulate executing the instructions and getting a new observation
        # TODO: Replace with real execution (e.g., run commands, collect output)
        observation = f"Processed observation from iteration {iteration}"

        # Update the state for the next iteration
        memory = updated_memory
        last_plan = new_plan
        last_action = "executed_instructions"  # Could parse from execution_instructions

        # Display the results
        console.print(f"Memory: {memory}", style="blue")
        console.print(f"Plan: {new_plan}", style="green")
        console.print(f"Execution Instructions: {execution_instructions}", style="yellow")
        
        # Process operations
        observation = ""
        for op in operations:
            if op[0] == "message":
                console.print(f"Message: {op[1]}", style="cyan")
            elif op[0] == "command":
                console.print(f"Would execute command: {op[1]}", style="magenta")
                # TODO: Actually execute the command and capture output
                observation += f"Command '{op[1]}' would return some output.\n"
            elif op[0] == "file":
                console.print(f"Would write to file {op[1]}", style="magenta")
                # TODO: Actually write to the file
                observation += f"File '{op[1]}' would be written.\n"
        
        if not observation:
            observation = f"Processed observation from iteration {iteration}"

    console.print("\nAgent run completed", style="bold green")

if __name__ == "__main__":
    app()
