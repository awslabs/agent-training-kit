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
def cancel_reservation(reservation_id: str) -> str:
    """
    Cancel the whole reservation. Only allowed for:
    - Reservations within 24 hours of booking
    - Business class reservations
    - Reservations with travel insurance
    
    Args:
        reservation_id: The reservation ID, such as 'ZFA04Y'.
    
    Returns:
        JSON string with the updated reservation showing cancelled status.
    """
    try:
        result = _get_wrapper().toolkit.cancel_reservation(reservation_id)
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})