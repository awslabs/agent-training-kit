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
def cancel_pending_order(order_id: str, reason: str) -> str:
    """
    Cancel a pending order. Only pending orders can be cancelled.
    Must get explicit user confirmation before calling.
    
    Args:
        order_id: The order ID, such as '#W0000000'.
        reason: Reason for cancellation - 'no longer needed' or 'ordered by mistake'.
    
    Returns:
        JSON string with updated order details showing cancelled status.
    """
    try:
        result = _get_wrapper().toolkit.cancel_pending_order(order_id, reason)
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
