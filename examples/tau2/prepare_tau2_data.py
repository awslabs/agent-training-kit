"""
Prepare Tau-2 dataset for agent training.
- Generates tickets from user_scenario.instructions
- Filters tasks suitable for solo mode
- Train on airline+retail train split, test on airline+retail test split
"""

import json
import random
from typing import Optional
from datasets import Dataset, disable_caching

from rllm.data.dataset import DatasetRegistry

import re

def convert_to_third_person(text: str) -> str:
    """
    Convert second-person pronouns (you/your/yours) to third-person (the customer/customer's).
    """
    # Order matters - do longer/specific patterns first to avoid partial replacements
    replacements = [
        # Fix "You [noun] is" patterns (e.g., "You name is" -> "The customer's name is")
        (r'\bYou (\w+) is\b', r"The customer's \1 is"),
        (r'\byou (\w+) is\b', r"the customer's \1 is"),
        
        # Fix "You [noun] are" patterns (e.g., "You orders are" -> "The customer's orders are")
        (r'\bYou (\w+) are\b', r"The customer's \1 are"),
        (r'\byou (\w+) are\b', r"the customer's \1 are"),
        
        # "You are" -> "The customer is"
        (r'\bYou are\b', 'The customer is'),
        (r'\byou are\b', 'the customer is'),
        (r'\bYou\'re\b', 'The customer is'),
        (r'\byou\'re\b', 'the customer is'),
        
        # "You were" -> "The customer was"
        (r'\bYou were\b', 'The customer was'),
        (r'\byou were\b', 'the customer was'),
        
        # "You have" -> "The customer has"
        (r'\bYou have\b', 'The customer has'),
        (r'\byou have\b', 'the customer has'),
        (r'\bYou\'ve\b', 'The customer has'),
        (r'\byou\'ve\b', 'the customer has'),
        
        # "You had" -> "The customer had"
        (r'\bYou had\b', 'The customer had'),
        (r'\byou had\b', 'the customer had'),
        
        # "You want" -> "The customer wants"
        (r'\bYou want\b', 'The customer wants'),
        (r'\byou want\b', 'the customer wants'),
        
        # "You need" -> "The customer needs"
        (r'\bYou need\b', 'The customer needs'),
        (r'\byou need\b', 'the customer needs'),
        
        # "You like" -> "The customer likes"
        (r'\bYou like\b', 'The customer likes'),
        (r'\byou like\b', 'the customer likes'),
        
        # "You know" -> "The customer knows"
        (r'\bYou know\b', 'The customer knows'),
        (r'\byou know\b', 'the customer knows'),
        
        # "You would" -> "The customer would"
        (r'\bYou would\b', 'The customer would'),
        (r'\byou would\b', 'the customer would'),
        (r'\bYou\'d\b', 'The customer would'),
        (r'\byou\'d\b', 'the customer would'),
        
        # "You will" -> "The customer will"
        (r'\bYou will\b', 'The customer will'),
        (r'\byou will\b', 'the customer will'),
        (r'\bYou\'ll\b', 'The customer will'),
        (r'\byou\'ll\b', 'the customer will'),
        
        # "You can" -> "The customer can"
        (r'\bYou can\b', 'The customer can'),
        (r'\byou can\b', 'the customer can'),
        
        # "You could" -> "The customer could"
        (r'\bYou could\b', 'The customer could'),
        (r'\byou could\b', 'the customer could'),
        
        # "You should" -> "The customer should"
        (r'\bYou should\b', 'The customer should'),
        (r'\byou should\b', 'the customer should'),
        
        # "You must" -> "The customer must"
        (r'\bYou must\b', 'The customer must'),
        (r'\byou must\b', 'the customer must'),
        
        # "You may" -> "The customer may"
        (r'\bYou may\b', 'The customer may'),
        (r'\byou may\b', 'the customer may'),
        
        # "You might" -> "The customer might"
        (r'\bYou might\b', 'The customer might'),
        (r'\byou might\b', 'the customer might'),
        
        # "You don't" -> "The customer doesn't"
        (r'\bYou don\'t\b', 'The customer doesn\'t'),
        (r'\byou don\'t\b', 'the customer doesn\'t'),
        (r'\bYou do not\b', 'The customer does not'),
        (r'\byou do not\b', 'the customer does not'),
        
        # "You didn't" -> "The customer didn't"
        (r'\bYou didn\'t\b', 'The customer didn\'t'),
        (r'\byou didn\'t\b', 'the customer didn\'t'),
        (r'\bYou did not\b', 'The customer did not'),
        (r'\byou did not\b', 'the customer did not'),
        
        # "You just" -> "The customer just"
        (r'\bYou just\b', 'The customer just'),
        (r'\byou just\b', 'the customer just'),
        
        # "You also" -> "The customer also"
        (r'\bYou also\b', 'The customer also'),
        (r'\byou also\b', 'the customer also'),
        
        # "You only" -> "The customer only"
        (r'\bYou only\b', 'The customer only'),
        (r'\byou only\b', 'the customer only'),
        
        # Possessives - do before general "you"
        (r'\bYours\b', "The customer's"),
        (r'\byours\b', "the customer's"),
        (r'\bYour\b', "The customer's"),
        (r'\byour\b', "the customer's"),
        
        # "Yourself" -> "themselves"
        (r'\bYourself\b', 'Themselves'),
        (r'\byourself\b', 'themselves'),
        
        # General "you" (as object, e.g., "contact you" -> "contact the customer")
        # These should come last
        
        # At start of sentence (after period, exclamation, question mark)
        (r'(?<=[.!?]\s)You\b', 'The customer'),
        (r'^You\b', 'The customer'),
        
        # Mid-sentence "you"
        (r'\bYou\b', 'the customer'),
        (r'\byou\b', 'the customer'),
    ]
    
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    
    return result

def remove_user_paragraph(text):
    pattern = r'You should transfer the user to a human agent if and only if[^\n]*(?:\n(?!\n)[^\n]*)*\n*'
    result = re.sub(pattern, '', text)
    pattern2 = r'Before taking any action[^\n]*(?:\n(?!\n)[^\n]*)*\n*'
    result2 = re.sub(pattern2, '', result)
    
    return result2


def generate_ticket_from_user_scenario(task: dict) -> Optional[str]:
    """
    Generate a ticket from user_scenario.instructions for solo mode.
    Returns None if insufficient information to generate ticket.
    """
    user_scenario = task.get("user_scenario")
    if not user_scenario:
        return None
    
    instructions = user_scenario.get("instructions")
    if not instructions:
        return None
    
    # Handle both string and structured instructions
    if isinstance(instructions, str):
        converted = convert_to_third_person(instructions)
        return f"Customer Issue:\n{converted}"
    
    # Structured instructions (StructuredUserInstructions)
    if isinstance(instructions, dict):
        parts = []
        
        # Known info (customer identity)
        known_info = instructions.get("known_info")
        if known_info:
            converted_info = convert_to_third_person(known_info)
            parts.append(f"Customer Information:\n{converted_info}")
        
        # Reason for call
        reason = instructions.get("reason_for_call")
        if reason:
            converted_reason = convert_to_third_person(reason)
            parts.append(f"Issue:\n{converted_reason}")
        
        # Additional details if present
        details = instructions.get("details")
        if details:
            converted_details = convert_to_third_person(details)
            parts.append(f"Details:\n{converted_details}")
        
        if parts:
            return "\n\n".join(parts)
    
    return None


def check_valid_task_for_solo(task: dict) -> bool:
    """
    Check if task is valid for solo mode (similar to LLMSoloAgent.check_valid_task).
    Task needs evaluation_criteria with actions.
    """
    eval_criteria = task.get("evaluation_criteria")
    if not eval_criteria:
        return False
    
    actions = eval_criteria.get("actions")
    if not actions or len(actions) == 0:
        return False
    
    return True

def get_policy_full(domain: str) -> str:
    """Get the full domain policy."""
    import os
    
    tau2_data_dir = os.environ.get("TAU2_DATA_DIR", "data/tau2/domains")
    policy_file = os.path.join(tau2_data_dir, domain, "policy.md")
    
    if os.path.exists(policy_file):
        with open(policy_file, "r") as f:
            content = f.read()
        return remove_user_paragraph(content)
    else:
        raise Exception(f"Fail to find policy file {policy_file}")

def load_tau2_tasks(domain: str) -> list[dict]:
    """Load tasks from Tau-2 data files."""
    import os
    
    # Find the tau2 data directory
    # Assuming tau2 data is in a known location or environment variable
    tau2_data_dir = os.environ.get("TAU2_DATA_DIR", "data/tau2/domains")
    
    tasks_file = os.path.join(tau2_data_dir, domain, "tasks.json")
    
    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    
    return tasks


def load_tau2_split(domain: str) -> dict:
    """Load split_tasks.json for a domain."""
    import os
    
    tau2_data_dir = os.environ.get("TAU2_DATA_DIR", "data/tau2/domains")
    split_file = os.path.join(tau2_data_dir, domain, "split_tasks.json")
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, "r") as f:
        split = json.load(f)
    
    return split


def load_tau2_db(domain: str) -> dict:
    """Load database from Tau-2 data files."""
    import os
    
    tau2_data_dir = os.environ.get("TAU2_DATA_DIR", "data/tau2/domains")
    db_file = os.path.join(tau2_data_dir, domain, "db.json")
    
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file not found: {db_file}")
    
    with open(db_file, "r") as f:
        db = json.load(f)
    
    return db


def prepare_tau2_data(
    domains: list[str] = ["airline", "retail"],
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = 42
):
    """
    Prepare Tau-2 dataset for training.
    
    Args:
        domains: List of domains to include (default: airline and retail)
        train_size: Optional limit on training examples
        test_size: Optional limit on test examples
    """
    
    def process_domain_tasks(domain: str, split_ids: list[str], max_size: Optional[int] = None) -> list[dict]:
        """Process tasks for a domain into training format, filtered by split IDs."""
        tasks = load_tau2_tasks(domain)
        policy_full = get_policy_full(domain)
        
        # Create a set of IDs for fast lookup
        split_ids_set = set(split_ids)
        
        processed = []
        for idx, task in enumerate(tasks):
            # Check if task is in the split
            if task["id"] not in split_ids_set:
                continue
            
            # Check if valid for solo mode
            if not check_valid_task_for_solo(task):
                continue
            
            # Generate ticket
            ticket = task.get("ticket") or generate_ticket_from_user_scenario(task)
            if not ticket:
                continue
            
            # Build the question (system context + ticket)
            question = f"""You are a customer service agent. Your task is to resolve the customer's issue by taking appropriate actions using the available tools.

## EVALUATION CRITERIA
Your performance is scored based on:
1. Calling the correct tools in the correct order
2. Passing the correct arguments to each tool

Unnecessary tool calls will reduce your score. Focus on making exactly the right tool calls with the right arguments.

## DOMAIN POLICY
<policy>
{policy_full}
</policy>

## CUSTOMER TICKET
<ticket>
{ticket}
</ticket>

## INSTRUCTIONS
Analyze the ticket and take the necessary actions to resolve the customer's issue. When finished, respond directly to the customer without calling any tool."""

            # Store evaluation criteria as ground truth
            eval_criteria = task.get("evaluation_criteria", {})
            initial_state = task.get("initial_state")
            
            processed.append({
                "question": question,
                "ground_truth": json.dumps(eval_criteria),
                "data_source": f"tau2_{domain}",
                "domain": domain,
                "uid": f"tau2_{domain}_{task['id']}",
                "task_id": task["id"],
                "index": idx,
                "initial_state": json.dumps(initial_state) if initial_state else None,
                "ticket": ticket,
                "policy_summary": policy_full,
            })
            
            if max_size and len(processed) >= max_size:
                break
        
        return processed
    
    # Collect train and test examples from all domains
    train_examples = []
    test_examples = []
    
    for domain in domains:
        split = load_tau2_split(domain)
        train_ids = split.get("train", [])
        test_ids = split.get("test", [])
        
        domain_train = process_domain_tasks(domain, train_ids)
        domain_test = process_domain_tasks(domain, test_ids)
        
        train_examples.extend(domain_train)
        test_examples.extend(domain_test)
        
        print(f"Domain {domain}: {len(domain_train)} train, {len(domain_test)} test")
    
    # Shuffle the combined examples
    random.seed(seed)
    random.shuffle(train_examples)
    random.shuffle(test_examples)
    # Apply size limits if specified
    if train_size and len(train_examples) > train_size:
        train_examples = train_examples[:train_size]
    if test_size and len(test_examples) > test_size:
        test_examples = test_examples[:test_size]
    
    print(f"Total: {len(train_examples)} training examples, {len(test_examples)} test examples")
    
    if len(train_examples) == 0:
        raise ValueError(f"No valid training examples found for domains: {domains}")
    if len(test_examples) == 0:
        raise ValueError(f"No valid test examples found for domains: {domains}")
    
    # Create datasets
    disable_caching()
    
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("tau2", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("tau2", test_dataset, "test")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    """
    # Default: airline + retail
    python prepare_tau2_data.py

    # Specify domains explicitly
    python prepare_tau2_data.py --domains airline retail

    # Single domain only
    python prepare_tau2_data.py --domains airline
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["airline", "retail"], 
                        choices=["airline", "retail", "telecom"])
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_dataset, test_dataset = prepare_tau2_data(
        domains=args.domains,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    
    print(f"\n=== Dataset Summary ===")
    print(f"Train dataset: {len(train_dataset.get_data())} examples from {args.domains}")
    print(f"Test dataset: {len(test_dataset.get_data())} examples from {args.domains}")
    
    print(f"\n=== Sample Training Example ===")
    sample = train_dataset.get_data()[0]
    print(f"UID: {sample['uid']}")
    print(f"Domain: {sample['domain']}")
    print(f"Question (truncated): {sample['question'][:500]}...")