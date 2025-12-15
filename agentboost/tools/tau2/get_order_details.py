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
def get_order_details(order_id: str) -> str:
    """
    Get the status and details of an order.
    
    Args:
        order_id: The order ID with '#' prefix, such as '#W0000000'.
    
    Returns:
        JSON string with order details including items, status,
        fulfillment info, and payment history.
    """
    try:
        result = _get_wrapper().toolkit.get_order_details(order_id)
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
