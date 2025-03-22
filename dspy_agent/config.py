import dspy

def configure_lm(model_name: str = "deepseek/deepseek-chat") -> None:
    """Centralized language model configuration"""
    # Special case for "flash" to use OpenRouter Gemini Flash
    if model_name.lower() == "flash":
        model_name = "openrouter/google/gemini-2.0-flash-001"
    
    lm = dspy.LM(model_name, 
            cache=False, 
            max_tokens=1000,
            )
    dspy.settings.configure(lm=lm)
