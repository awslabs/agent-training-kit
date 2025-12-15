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
def send_certificate(user_id: str, amount: int) -> str:
    """
    Send a certificate/voucher to a user as compensation.
    Use for flight delays, cancellations, or service issues.
    
    Args:
        user_id: The ID of the user, such as 'sara_doe_496'.
        amount: The amount of the certificate in dollars.
    
    Returns:
        Confirmation message with certificate details.
    """
    try:
        result = _get_wrapper().toolkit.send_certificate(user_id, amount)
        return result
    except Exception as e:
        return json.dumps({"error": str(e)})