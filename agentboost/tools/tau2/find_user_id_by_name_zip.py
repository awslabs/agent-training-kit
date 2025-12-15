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
def find_user_id_by_name_zip(first_name: str, last_name: str, zip_code: str) -> str:
    """
    Find user id by first name, last name, and zip code.
    
    Args:
        first_name: The first name, such as 'John'.
        last_name: The last name, such as 'Doe'.
        zip_code: The zip code, such as '12345'.
    
    Returns:
        The user ID if found, or error message.
    """
    try:
        return _get_wrapper().toolkit.find_user_id_by_name_zip(first_name, last_name, zip_code)
    except Exception as e:
        return json.dumps({"error": str(e)})
