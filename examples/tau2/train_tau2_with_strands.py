"""
Train agent on Tau-2 benchmark using Strands.
Phase 1: Train on airline, test on retail.
"""

import hydra
import json
from typing import Dict, Any, List

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from agentboost.agents.strands_agent import StrandsAgent
from agentboost.environments.tau2_env import Tau2Env
from agentboost.environments.tau2_reward import Tau2RewardFunction

from agentboost.tools.tau2 import find_user_id_by_email
from agentboost.tools.tau2 import find_user_id_by_name_zip
from agentboost.tools.tau2 import get_user_details
from agentboost.tools.tau2 import get_order_details
from agentboost.tools.tau2 import get_product_details
from agentboost.tools.tau2 import list_all_product_types
from agentboost.tools.tau2 import cancel_pending_order
from agentboost.tools.tau2 import modify_pending_order_items
from agentboost.tools.tau2 import modify_pending_order_address
from agentboost.tools.tau2 import modify_pending_order_payment
from agentboost.tools.tau2 import modify_user_address
from agentboost.tools.tau2 import return_delivered_order_items
from agentboost.tools.tau2 import exchange_delivered_order_items
from agentboost.tools.tau2 import calculate
#from agentboost.tools.tau2 import transfer_to_human_agents
#from agentboost.tools.tau2 import done

from agentboost.tools.tau2 import book_reservation
from agentboost.tools.tau2 import cancel_reservation
from agentboost.tools.tau2 import get_reservation_details
from agentboost.tools.tau2 import search_direct_flight
from agentboost.tools.tau2 import search_onestop_flight
from agentboost.tools.tau2 import list_all_airports
from agentboost.tools.tau2 import get_flight_status
from agentboost.tools.tau2 import update_reservation_flights
from agentboost.tools.tau2 import update_reservation_passengers
from agentboost.tools.tau2 import update_reservation_baggages
from agentboost.tools.tau2 import send_certificate

# Tools list:
RETAIL_TOOLS = [
    find_user_id_by_email,
    find_user_id_by_name_zip,
    get_user_details,
    get_order_details,
    get_product_details,
    list_all_product_types,
    cancel_pending_order,
    modify_pending_order_items,
    modify_pending_order_address,
    modify_pending_order_payment,
    modify_user_address,
    return_delivered_order_items,
    exchange_delivered_order_items,
    calculate,
    #transfer_to_human_agents,
]

AIRLINE_TOOLS = [
    get_reservation_details,
    search_direct_flight,
    search_onestop_flight,
    list_all_airports,
    get_flight_status,
    book_reservation,
    cancel_reservation,
    update_reservation_flights,
    update_reservation_passengers,
    update_reservation_baggages,
    send_certificate,
    #transfer_to_human_agents,
]
ALL_TOOLS = RETAIL_TOOLS + AIRLINE_TOOLS
#ALL_TOOLS = list(set(RETAIL_TOOLS + AIRLINE_TOOLS))

SYSTEM_PROMPT = """You are an autonomous customer service agent.

Execute actions directly WITHOUT ASKING for user confirmation or clarifying questions.
"""


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main training function."""
    
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("tau2", "train")
    test_dataset = DatasetRegistry.load_dataset("tau2", "test")
    
    agent_args = {
        "tools": ALL_TOOLS,
        "system_prompt": SYSTEM_PROMPT,
    }
    
    env_args = {
        "reward_fn": Tau2RewardFunction(),
    }
    
    trainer = AgentTrainer(
        agent_class=StrandsAgent,
        env_class=Tau2Env,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()