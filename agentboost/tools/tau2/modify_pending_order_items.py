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
def modify_pending_order_items(order_id: str, item_ids: str, new_item_ids: str, payment_method_id: str) -> str:
    """
    Modify items in a pending order to different variants of the SAME product.
    Can only be called once per order. Must get user confirmation first.
    
    Args:
        order_id: The order ID, such as '#W0000000'.
        item_ids: JSON array of item IDs to modify, such as '["1008292230"]'.
        new_item_ids: JSON array of new item IDs (same order as item_ids).
        payment_method_id: Payment method for price difference.
    
    Returns:
        JSON string with updated order details.
    """
    try:
        item_ids_list = json.loads(item_ids) if isinstance(item_ids, str) else item_ids
        new_item_ids_list = json.loads(new_item_ids) if isinstance(new_item_ids, str) else new_item_ids
        result = _get_wrapper().toolkit.modify_pending_order_items(
            order_id=order_id,
            item_ids=item_ids_list,
            new_item_ids=new_item_ids_list,
            payment_method_id=payment_method_id
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
