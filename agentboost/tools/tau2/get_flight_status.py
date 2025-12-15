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
def get_flight_status(flight_number: str, date: str) -> str:
    """
    Get the status of a flight.
    
    Args:
        flight_number: The flight number, such as 'HAT001'.
        date: The date of the flight in the format 'YYYY-MM-DD'.
    
    Returns:
        Status string: 'available', 'on time', 'delayed', 'cancelled', 'landed', or 'flying'.
    """
    try:
        result = _get_wrapper().toolkit.get_flight_status(flight_number, date)
        return result
    except Exception as e:
        return json.dumps({"error": str(e)})