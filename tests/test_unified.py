from dspy_agent.unified import UnifiedModule

def test_unified_module_initialization():
    module = UnifiedModule()
    assert module.predictor is not None, "Predictor should be initialized"

def test_xml_validation():
    module = UnifiedModule()
    valid_xml = "<agent_output><updated_memory>test</updated_memory></agent_output>"
    is_valid, _ = module.validate_xml(valid_xml)
    assert is_valid, "Should validate good XML"
