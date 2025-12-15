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
def get_reservation_details(reservation_id: str) -> str:
    """
    Get the details of a reservation.
    
    Args:
        reservation_id: The reservation ID, such as '8JX2WO'.
    
    Returns:
        JSON string with reservation details including flights, passengers,
        cabin class, baggage, insurance, and payment history.
    """
    try:
        result = _get_wrapper().toolkit.get_reservation_details(reservation_id)
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})