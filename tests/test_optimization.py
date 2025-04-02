from dspy_agent.optimization import Optimizer

def test_optimizer_initialization():
    optimizer = Optimizer(model_name="test-model", optimizer_type="bootstrap")
    assert optimizer.optimizer is not None, "Optimizer should be initialized"

def test_bootstrap_optimization(tmp_path):
    # Create simple training data
    test_data = tmp_path / "test_data.jsonl"
    test_data.write_text("""{"input_xml": "<test/>", "output_xml": "<test/>"}""")
    
    optimizer = Optimizer(optimizer_type="bootstrap")
    optimized_module = optimizer.optimize(str(test_data))
    
    assert optimized_module is not None, "Should return optimized module"
