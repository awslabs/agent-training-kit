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
def find_user_id_by_email(email: str) -> str:
    """
    Find user id by email. Preferred method for user lookup.
    
    Args:
        email: The email of the user, such as 'john@example.com'.
    
    Returns:
        The user ID if found, or error message.
    """
    try:
        return _get_wrapper().toolkit.find_user_id_by_email(email)
    except Exception as e:
        return json.dumps({"error": str(e)})
