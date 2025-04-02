from dspy_agent.unified import UnifiedModule

def test_unified_module_initialization():
    module = UnifiedModule()
    assert module.predictor is not None, "Predictor should be initialized"

def test_xml_validation():
    module = UnifiedModule()
    valid_xml = """
    <updated_memory>test</updated_memory>
    <new_plan><plan><goal>Test</goal><steps><step id="1"><action>Test</action></step></steps></plan></new_plan>
    <execution_instructions><write_operations><operation type="message">Test</operation></write_operations></execution_instructions>
    <expected_outcome>Test</expected_outcome>
    <is_done>false</is_done>
    """
    is_valid, _ = module.validate_xml(valid_xml)
    assert is_valid, "Should validate good XML"
