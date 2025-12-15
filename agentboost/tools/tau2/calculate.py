from strands import tool
from agentboost.tools.tau2_retail_tools import RetailToolsWrapper
import json

_wrapper = None


def _get_wrapper():
    global _wrapper
    if _wrapper is None:
        _wrapper = RetailToolsWrapper()
    return _wrapper


@tool
def calculate(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        expression: The mathematical expression, such as '2 + 2' or '(100 - 50) * 2'.
    
    Returns:
        The calculated result as a string.
    """
    try:
        return _get_wrapper().toolkit.calculate(expression)
    except Exception as e:
        return json.dumps({"error": str(e)})
