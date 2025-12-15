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
def update_reservation_baggages(
    reservation_id: str,
    total_baggages: int,
    nonfree_baggages: int,
    payment_id: str
) -> str:
    """
    Update the baggage information of a reservation.
    Can add bags but cannot remove bags.
    
    Args:
        reservation_id: The reservation ID, such as 'ZFA04Y'.
        total_baggages: The updated total number of baggage items.
        nonfree_baggages: The updated number of paid baggage items.
        payment_id: Payment method ID for baggage fees, such as 'credit_card_7815826'.
    
    Returns:
        JSON string with the updated reservation details.
    """
    try:
        result = _get_wrapper().toolkit.update_reservation_baggages(
            reservation_id=reservation_id,
            total_baggages=total_baggages,
            nonfree_baggages=nonfree_baggages,
            payment_id=payment_id
        )
        return json.dumps(result.model_dump(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})