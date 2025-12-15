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
def book_reservation(
    user_id: str,
    origin: str,
    destination: str,
    flight_type: str,
    cabin: str,
    flights: str,
    passengers: str,
    payment_methods: str,
    total_baggages: int,
    nonfree_baggages: int,
    insurance: str
) -> str:
    """
    Book a reservation.
    
    Args:
        user_id: The ID of the user, such as 'sara_doe_496'.
        origin: The IATA code for the origin city, such as 'SFO'.
        destination: The IATA code for the destination city, such as 'JFK'.
        flight_type: Type of flight - 'one_way' or 'round_trip'.
        cabin: Cabin class - 'basic_economy', 'economy', or 'business'.
        flights: JSON string array of flight objects with 'flight_number' and 'date'.
        passengers: JSON string array of passenger objects with 'first_name', 'last_name', 'dob'.
        payment_methods: JSON string array of payment objects with 'payment_id' and 'amount'.
        total_baggages: Total number of baggage items.
        nonfree_baggages: Number of paid baggage items.
        insurance: Whether to include insurance - 'yes' or 'no'.
    
    Returns:
        JSON string with the created reservation details.
    """
    try:
        flights_list = json.loads(flights) if isinstance(flights, str) else flights
        passengers_list = json.loads(passengers) if isinstance(passengers, str) else passengers
        payments_list = json.loads(payment_methods) if isinstance(payment_methods, str) else payment_methods
        
        result = _get_wrapper().toolkit.book_reservation(
            user_id=user_id,
            origin=origin,
            destination=destination,
            flight_type=flight_type,
            cabin=cabin,
            flights=flights_list,
            passengers=passengers_list,
            payment_methods=payments_list,
            total_baggages=total_baggages,
            nonfree_baggages=nonfree_baggages,
            insurance=insurance
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})