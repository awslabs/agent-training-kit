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
def modify_pending_order_address(order_id: str, address1: str, address2: str, city: str, state: str, country: str, zip_code: str) -> str:
    """
    Modify the shipping address of a pending order.
    Must get user confirmation first.
    
    Args:
        order_id: The order ID, such as '#W0000000'.
        address1: Primary address line, such as '123 Main St'.
        address2: Secondary address line, such as 'Apt 1' or empty string.
        city: City name, such as 'San Francisco'.
        state: State code, such as 'CA'.
        country: Country name, such as 'USA'.
        zip_code: Postal code, such as '12345'.
    
    Returns:
        JSON string with updated order details.
    """
    try:
        result = _get_wrapper().toolkit.modify_pending_order_address(
            order_id=order_id,
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
