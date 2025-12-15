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
def exchange_delivered_order_items(order_id: str, item_ids: str, new_item_ids: str, payment_method_id: str) -> str:
    """
    Exchange items in a delivered order for different variants of the SAME product.
    For delivered orders, return or exchange can only be done once.
    Must get user confirmation first.
    
    Args:
        order_id: The order ID, such as '#W0000000'.
        item_ids: JSON array of item IDs to exchange.
        new_item_ids: JSON array of new item IDs (same order, same products).
        payment_method_id: Payment method for price difference.
    
    Returns:
        JSON string with updated order showing 'exchange requested' status.
    """
    try:
        item_ids_list = json.loads(item_ids) if isinstance(item_ids, str) else item_ids
        new_item_ids_list = json.loads(new_item_ids) if isinstance(new_item_ids, str) else new_item_ids
        result = _get_wrapper().toolkit.exchange_delivered_order_items(
            order_id=order_id,
            item_ids=item_ids_list,
            new_item_ids=new_item_ids_list,
            payment_method_id=payment_method_id
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
