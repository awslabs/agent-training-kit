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
def list_all_product_types() -> str:
    """
    List the name and product id of all product types.
    There are 50 product types in the store.
    
    Returns:
        JSON string mapping product names to product IDs.
    """
    try:
        return _get_wrapper().toolkit.list_all_product_types()
    except Exception as e:
        return json.dumps({"error": str(e)})
