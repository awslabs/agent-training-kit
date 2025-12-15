from strands import tool
from agentboost.tools.tau2_retail_tools import RetailToolsWrapper
import json

_wrapper = None


def _get_wrapper():
    global _wrapper
    if _wrapper is None:
        _wrapper = RetailToolsWrapper()
    return _wrapper


@tool
def transfer_to_human_agents(summary: str) -> str:
    """
    Transfer the user to a human agent. Only use when:
    - The user explicitly asks for a human agent
    - The issue cannot be resolved with available tools
    
    Args:
        summary: A summary of the user's issue for the human agent.
    
    Returns:
        Confirmation of transfer.
    """
    try:
        return _get_wrapper().toolkit.transfer_to_human_agents(summary)
    except Exception as e:
        return json.dumps({"error": str(e)})
