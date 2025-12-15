from strands import tool
from agentboost.tools.tau2_retail_tools import RetailToolsWrapper
from agentboost.tools.tau2_airline_tools import AirlineToolsWrapper
import json

_wrapper_retail = None
_wrapper_airline = None

def _get_wrapper(domain="retail"):
    global _wrapper_retail
    global _wrapper_airline
    if domain == "retail":
        if _wrapper_retail is None:
            _wrapper_retail = RetailToolsWrapper()
        return _wrapper_retail
    elif domain == "airline":
        if _wrapper_airline is None:
            _wrapper_airline = AirlineToolsWrapper()
        return _wrapper_airline
    else:
        return None

@tool
def get_user_details(user_id: str) -> str:
    """
    Get the details of a user, including their orders.
    
    Args:
        user_id: The user ID, such as 'sara_doe_496'.
    
    Returns:
        JSON string with user details including name, address, email,
        payment methods, and order IDs.
    """
    try:
        result = _get_wrapper().toolkit.get_user_details(user_id)
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        if "User not found" in str(e):
            try:
                result = _get_wrapper(domain="airline").toolkit.get_user_details(user_id)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e1:
                return json.dumps({"error": str(e1)})
        else:
            return json.dumps({"error": str(e)})
