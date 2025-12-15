"""
Environment for Tau-2 benchmark tasks.
Handles tool execution and evaluation using Tau-2's built-in evaluation system.
"""

import json
import warnings
from typing import Any, Dict, List, Optional
from copy import deepcopy

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, RewardOutput

from agentboost.environments.tau2_reward import Tau2RewardFunction as Tau2RewardFunction_FG

class Tau2Env(BaseEnv):
    """
    Environment for Tau-2 benchmark tasks.
    """
    
    def __init__(
        self,
        task: Optional[Dict] = None,
        reward_fn: Optional[RewardFunction] = None,
        max_steps: int = 10,
        validation: bool = False,
        domain: str = "airline",
    ):
        """
        Initialize the Tau2 environment.
        
        Args:
            task: Task information including question, ground_truth, etc.
            reward_fn: Reward function (defaults to Tau2RewardFunction)
            max_steps: Maximum number of tool-calling steps
            validation: Whether running in validation mode
            domain: Domain name ('airline' or 'retail')
        """
        self.step_count = 0
        self.max_steps = max_steps
        self.task = task or {}
        self.validation = validation
        self.domain = domain
        
        if reward_fn is None:
            self.reward_fn = Tau2RewardFunction_FG()
        else:
            self.reward_fn = reward_fn
        
        # Track tool calls made during episode
        self.tool_calls_made: List[Dict] = []
        
        # Database state tracking
        self._initial_db_state = None
        self._current_db_state = None
    
    def reset(self):
        """Reset the environment for a new episode."""
        self.step_count = 0
        self.tool_calls_made = []
        
        # Extract domain from task if available
        if self.task:
            self.domain = self.task.get("domain", self.domain)
        
        return self.task, {}
    
    def step(self, action: Any):
        """
        Process an action (tool call or completion).
        
        Args:
            action: Either a string response, dict tool call, or list of tool calls
        
        Returns:
            (next_obs, reward, done, info)
        """
        if action is None:
            action = []
        
        if isinstance(action, dict):
            action = [action]
        
        self.step_count += 1
        reward = 0.0
        
        # Check for completion
        done = self.step_count >= self.max_steps or isinstance(action, str)
        final_response = None

        if isinstance(action, list) and action:
            for tool_call in action:
                if "function" in tool_call:
                    func_info = tool_call["function"]
                    func_name = func_info.get("name", "")
                    args = func_info.get("arguments", {})
                else:
                    func_name = tool_call.get("name", "")
                    args = tool_call.get("arguments", {})
                
                # Parse args first
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                
                # Check for strands_agent-added control FINISH tool
                if func_name == "finish" and "response" in args:
                    done = True
                    final_response = args["response"]
                    continue
                
                if func_name:
                    self.tool_calls_made.append({
                        "name": func_name,
                        "arguments": args,
                    })
        
        # Calculate final reward when done
        if done:
            # Get the final response
            if final_response is None:
                final_response = str(action)

            # Build task info for reward calculation
            task_info = {
                **self.task,
                "tool_calls_made": self.tool_calls_made,
                "domain": self.domain,
                "final_response": final_response,
            }
            
            reward_output = self.reward_fn(task_info=task_info, action=final_response)
            reward = reward_output.reward
            
            return {}, reward, done, {
                "response": action,
                "metadata": reward_output.metadata,
                "tool_calls_made": self.tool_calls_made,
            }
        
        # Not done yet, return observation for next step
        next_obs = {"tool_outputs": {}}
        
        return next_obs, reward, done, {
            "response": action,
            "metadata": {},
        }
    
    @staticmethod
    def from_dict(env_args: Dict) -> "Tau2Env":
        """Create environment from dictionary of arguments."""
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        validation = env_args.pop("validation", False)
        domain = env_args.pop("domain", "airline")
        
        return Tau2Env(
            task=env_args,
            reward_fn=reward_fn,
            max_steps=max_steps,
            validation=validation,
            domain=domain,
        )