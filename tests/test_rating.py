from dspy_agent.rating import RatingModule

def test_rating_module_returns_valid_score():
    rater = RatingModule()
    test_input = "<agent_state><observation>What is 2+2?</observation></agent_state>"
    test_output = "<agent_output><updated_memory>4</updated_memory></agent_output>"
    
    score = rater(test_input, test_output)
    
    assert 1 <= score <= 9, "Score should be between 1 and 9"
