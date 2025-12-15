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
def update_reservation_flights(
    reservation_id: str,
    cabin: str,
    flights: str,
    payment_id: str
) -> str:
    """
    Update the flight information of a reservation.
    Cannot change basic economy reservations. Cannot change origin/destination.
    
    Args:
        reservation_id: The reservation ID, such as 'ZFA04Y'.
        cabin: The cabin class - 'basic_economy', 'economy', or 'business'.
        flights: JSON string array of ALL flight objects in the new reservation.
        payment_id: Payment method ID for any price difference, such as 'credit_card_7815826'.
    
    Returns:
        JSON string with the updated reservation details.
    """
    try:
        flights_list = json.loads(flights) if isinstance(flights, str) else flights
        
        result = _get_wrapper().toolkit.update_reservation_flights(
            reservation_id=reservation_id,
            cabin=cabin,
            flights=flights_list,
            payment_id=payment_id
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})