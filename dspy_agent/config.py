import dspy

def configure_lm(model_name: str = "deepseek/deepseek-chat") -> None:
    """Centralized language model configuration"""
    lm = dspy.LM(model_name, cache=False)
    dspy.settings.configure(lm=lm)
