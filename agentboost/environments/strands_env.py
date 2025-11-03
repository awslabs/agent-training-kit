import warnings
from typing import Any

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward


class StrandsEnv(BaseEnv):
    """
    Environment for StrandsAgent
    """

    def __init__(self, task: dict | None = None, reward_fn: RewardFunction | None = None, max_steps=10, validation=False):
        """
        Initialize the StrandsEnv.

        Args:
            task: Task information for the environment.
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
            valiation: True if running in val/test mode 
        """
        self.step_count = 0
        self.max_steps = max_steps
        self.task = task
        
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward
        else:
            self.reward_fn = reward_fn
        self.validation = validation

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        return self.task, {}

    def step(self, action: list[dict] | str | dict):
        """
        Take a step in the environment based on the action.

        Args:
            action: Action from StrandsAgent (can be string or structured tool calls)

        Returns:
            next_observations, rewards, terminateds, infos
        """
        
        if action is None:
            action = []

        if isinstance(action, dict):
            action = [action]
        
        self.step_count += 1
        reward = 0
        
        done = self.step_count >= self.max_steps or isinstance(action, str)
        
        if isinstance(action, list) and action:
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True
                    break
        
        if done:
            if isinstance(action, str):
                llm_response = action
            elif isinstance(action, list):
                finish_action = None
                for tool_call in action:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call
                        break
                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {})
                    llm_response = arguments.get("response", "")
                else:
                    llm_response = str(action)

            task_info = self.task if self.task is not None else {}
            reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            if (self.validation):
                final_reward = reward_output.reward
            else:
                if "error" in reward_output.metadata:
                    if "No code found" in reward_output.metadata["error"]:
                        final_reward = -1.0
                    else:
                        final_reward = 0.0
                else:
                    total_tests = reward_output.metadata.get("total_tests", 0)
                    passed_tests = reward_output.metadata.get("passed_tests", 0)
                    if total_tests > 0:
                        final_reward = passed_tests / total_tests
                    else:
                        final_reward = 0.0
            return {}, final_reward, done, {"response": action, "metadata": reward_output.metadata}

        tool_calls = action
        assert isinstance(tool_calls, list)

        tool_reward = 0.0
        if (not self.validation) and tool_calls and isinstance(tool_calls[0], dict) and "reward" in tool_calls[0]:
            tool_reward = tool_calls[0]["reward"]
            reward = tool_reward * 0.2
            print(f"ToolCalling -- tool-reward = {reward} ")
        
        next_obs = {"tool_outputs": {}}
        
        return next_obs, reward, done, {"response": action, "metadata": {}}

    @staticmethod
    def from_dict(env_args: dict) -> "StrandsEnv":
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        validation = env_args.pop("validation", False)
        return StrandsEnv(task=env_args, reward_fn=reward_fn, max_steps=max_steps, validation=validation)