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
def search_direct_flight(origin: str, destination: str, date: str) -> str:
    """
    Search for direct flights between two cities on a specific date.
    
    Args:
        origin: The origin city airport in three letters, such as 'JFK'.
        destination: The destination city airport in three letters, such as 'LAX'.
        date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.
    
    Returns:
        JSON string with list of available direct flights including 
        flight numbers, times, prices, and available seats.
    """
    try:
        result = _get_wrapper().toolkit.search_direct_flight(origin, destination, date)
        return json.dumps([r.model_dump() for r in result], indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})