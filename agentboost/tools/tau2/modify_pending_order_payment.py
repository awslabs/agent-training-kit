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
def modify_pending_order_payment(order_id: str, payment_method_id: str) -> str:
    """
    Modify the payment method of a pending order.
    Must get user confirmation first.
    
    Args:
        order_id: The order ID, such as '#W0000000'.
        payment_method_id: New payment method ID.
    
    Returns:
        JSON string with updated order details.
    """
    try:
        result = _get_wrapper().toolkit.modify_pending_order_payment(
            order_id=order_id,
            payment_method_id=payment_method_id
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
