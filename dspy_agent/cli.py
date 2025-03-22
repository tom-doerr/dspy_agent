import typer
import json
import os
import subprocess
from rich.console import Console
import xml.etree.ElementTree as ET
import dspy
from .unified import UnifiedModule
from .schema import INPUT_XML_SCHEMA, OUTPUT_XML_SCHEMA

app = typer.Typer()
console = Console()

@app.command()
def generate_training_data(
    output_file: str = typer.Argument(..., help="Path to save training data"),
    count: int = typer.Option(10, help="Number of training examples to generate"),
    no_output: bool = typer.Option(False, "--no-output", help="Generate examples without output XML"),
):
    """Generate synthetic training data for optimization."""
    import random
    from .schema import EXAMPLE_INPUT_XML, EXAMPLE_OUTPUT_XML
    
    examples = []
    for _ in range(count):
        # Vary the examples
        input_xml = EXAMPLE_INPUT_XML.replace("previous_action", f"action_{random.randint(1,100)}") \
            .replace("Previous knowledge", f"Knowledge v{random.randint(1,100)}") \
            .replace("Result of the last action", f"Observation {random.randint(1,100)}")
            
        # Generate output XML unless disabled
        if no_output:
            output_xml = ""
        else:
            output_xml = EXAMPLE_OUTPUT_XML.replace("false", random.choice(["true", "false"])) \
                .replace("Find all Python files", random.choice([
                    "Analyze log files",
                    "Process user data",
                    "Generate report"
                ]))
        
        examples.append(dspy.Example(
            input_schema=INPUT_XML_SCHEMA,
            output_schema=OUTPUT_XML_SCHEMA,
            input_xml=input_xml,
            output_xml=output_xml
        ).with_inputs("input_schema", "output_schema", "input_xml"))
    
    # Save to JSONL
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps({
                "input_schema": INPUT_XML_SCHEMA,
                "output_schema": OUTPUT_XML_SCHEMA,
                "input_xml": ex.input_xml,
                "output_xml": ex.output_xml
            }) + "\n")
            
    console.print(f"Generated {count} training examples in {output_file}", style="bold green")

@app.command()
def optimize(
    training_data: str = typer.Argument(..., help="Path to training data file"),
    model: str = typer.Option("deepseek/deepseek-chat", help="The model to use"),
    optimizer: str = typer.Option(
        "bootstrap", 
        help="Optimizer to use: bootstrap, random_search, mipro"
    )
):
    """Optimize the DSPy module using training data."""
    from .optimizer import Optimizer
    
    try:
        optimizer = Optimizer(model_name=model, optimizer_type=optimizer)
        optimizer.optimize(training_data)
    except Exception as e:
        console.print(f"Optimization failed: {str(e)}", style="red")
        raise typer.Exit(code=1)

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
    from .config import configure_lm
    configure_lm(model)
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
        console.print(f"Input XML: {input_xml}", flush=True)
        output_xml = unified_module(input_xml)
        console.print(f"Output XML: {output_xml}")
        
        # Log validation status
        is_valid, error = unified_module.validate_xml(output_xml)
        if not is_valid:
            console.print(f"Warning: Output XML validation failed: {error}", style="yellow")

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
                console.print(f"\nExecuting: {op[1]}", style="bold magenta")
                console.print("WARNING: Executing arbitrary commands can be dangerous!", style="red")
                
                # Confirm execution
                confirm = input("Proceed with execution? [y/N] ").strip().lower()
                if confirm != 'y':
                    observation += f"Command execution aborted by user: {op[1]}\n"
                    continue
                
                try:
                    result = subprocess.run(
                        op[1], 
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    observation += f"Command output:\n{result.stdout}\n"
                    if result.stderr:
                        observation += f"Errors:\n{result.stderr}\n"
                except subprocess.TimeoutExpired:
                    observation += f"Command timed out: {op[1]}\n"
                except Exception as e:
                    observation += f"Command failed: {str(e)}\n"
            elif op[0] == "file":
                console.print(f"Would write to file {op[1]}", style="magenta")
                # TODO: Actually write to the file
                observation += f"File '{op[1]}' would be written.\n"
        
        if not observation:
            observation = f"Processed observation from iteration {iteration}"

    console.print("\nAgent run completed", style="bold green")

if __name__ == "__main__":
    app()
