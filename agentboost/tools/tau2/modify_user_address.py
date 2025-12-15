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
def modify_user_address(user_id: str, address1: str, address2: str, city: str, state: str, country: str, zip_code: str) -> str:
    """
    Modify the default address of a user.
    Must get user confirmation first.
    
    Args:
        user_id: The user ID, such as 'sara_doe_496'.
        address1: Primary address line.
        address2: Secondary address line or empty string.
        city: City name.
        state: State code.
        country: Country name.
        zip_code: Postal code.
    
    Returns:
        JSON string with updated user details.
    """
    try:
        result = _get_wrapper().toolkit.modify_user_address(
            user_id=user_id,
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip_code
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
