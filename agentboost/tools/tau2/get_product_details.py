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
def get_product_details(product_id: str) -> str:
    """
    Get the inventory details of a product.
    
    Args:
        product_id: The product ID, such as '6086499569'.
                   Note: product_id is different from item_id.
    
    Returns:
        JSON string with product name and all variants with 
        their options, availability, and prices.
    """
    try:
        result = _get_wrapper().toolkit.get_product_details(product_id)
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
