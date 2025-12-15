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
def update_reservation_passengers(reservation_id: str, passengers: str) -> str:
    """
    Update the passenger information of a reservation.
    Number of passengers must remain the same.
    
    Args:
        reservation_id: The reservation ID, such as 'ZFA04Y'.
        passengers: JSON string array of passenger objects with 'first_name', 'last_name', 'dob'.
    
    Returns:
        JSON string with the updated reservation details.
    """
    try:
        passengers_list = json.loads(passengers) if isinstance(passengers, str) else passengers
        
        result = _get_wrapper().toolkit.update_reservation_passengers(
            reservation_id=reservation_id,
            passengers=passengers_list
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})