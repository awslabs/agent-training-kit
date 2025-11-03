import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from agentboost.agents.strands_agent import StrandsAgent
from agentboost.tools import strands_code_tool
from agentboost.environments.strands_env import StrandsEnv

system_prompt = """You are an expert competitive programming assistant.

CRITICAL: When using strands_code_tool, NEVER put code in function parameters. 
Write code in ```python blocks in your response. The tool extracts code automatically.

WRONG: strands_code_tool(code="your code")  
RIGHT: Write ```python\ncode\n``` then call strands_code_tool()

**WARNING**: Your final submission MUST contain a ```python block. No code block = failed submission.
This can happen if you think too much and hit the 16K token response limit. Be concise.

## Your Process:
1. **Analyze** the problem - understand requirements, constraints, examples
2. **Write solution** in a ```python code block with proper input/output
3. **Test with strands_code_tool()** OR make final submission

## Tool Available:
`strands_code_tool()` - Extracts Python from your response and tests against samples
- Call with NO parameters: strands_code_tool()
- Automatically finds and tests your most recent ```python block

## Autonomous Decision Making:
- **ALWAYS test first**: Call strands_code_tool() after writing any solution
- **Tests pass → IMMEDIATE SUBMISSION**: Next response is ONLY code block (no text, no tool call)
- **Tests fail → Debug and retry, or submit only if confident

## Submission Rules:
**Tests passed or verified correct → Code only submission:**
```python
[your code]
```
THIS IS YOUR FINAL SUBMISSION - No explanations before/after

## Code Requirements:
- Read input with `input()` exactly as problem specifies
- Output with `print()` in exact required format
- Handle multiple test cases when problem has them

REMEMBER: Code in ```python blocks, NOT parameters. Keep responses concise to avoid token limit."""


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("agentcoder", "train")
    test_dataset = DatasetRegistry.load_dataset("agentcoder", "test")

    agent_args = {"tools": [strands_code_tool],
                  "system_prompt": system_prompt}
    env_args = {
        "reward_fn": code_reward_fn,
    }

    trainer = AgentTrainer(
        agent_class=StrandsAgent,
        env_class=StrandsEnv,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
