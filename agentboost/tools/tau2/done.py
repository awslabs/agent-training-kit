from strands import tool


@tool
def done(response: str) -> str:
    """
    Call this when you have completed all necessary actions for the customer's issue.
    
    Args:
        response: A summary of what actions were taken to resolve the issue.
    
    Returns:
        Confirmation that the task is complete.
    """
    return f"Task completed: {response}"
