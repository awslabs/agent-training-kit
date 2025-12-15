from strands import tool
from agentboost.tools.tau2_airline_tools import AirlineToolsWrapper
import json

_wrapper = None


def _get_wrapper():
    global _wrapper
    if _wrapper is None:
        _wrapper = AirlineToolsWrapper()
    return _wrapper


@tool
def list_all_airports() -> str:
    """
    Returns a list of all available airports.
    
    Returns:
        JSON string with list of airport codes and city names.
    """
    try:
        result = _get_wrapper().toolkit.list_all_airports()
        return json.dumps([r.model_dump() for r in result], indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})