# Tau-2 Benchmark Example

Adapts the [Tau-2 benchmark](https://github.com/sierra-research/tau2-bench) for agent training using [ToolRL](https://arxiv.org/abs/2504.13958)-style rewards.

## Dataset

- Converts interactive user-agent scenarios to single-turn ticket resolution tasks
- Transforms second-person instructions ("You are...") to third-person ("The customer is...")
- Filters for tasks with ground-truth tool call sequences (required for reward computation)
- Domains: [Airline](https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/domains/airline) (11 tools), [Retail](https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/domains/retail) (14 tools)
- Train: 97 examples | Test: 58 examples (combined and shuffled across domains)

## Reward Function

Based on ToolRL's three-level decomposition:

```
R_final = R_format + R_correct ∈ [-3, 4]
```

- **Format** (0-1): Valid tool call structure
- **Correctness** (-3 to 3): Tool name matching (Jaccard) + parameter key matching + value matching

The correctness reward has a larger scale than the format reward to encourage accurate tool usage over superficial formatting.

## Training Configurations

Two models have been tested. All [configs](./train_tau2_with_strands.sh) are shared except for the following:

| Model | actor_rollout_ref.model.path | actor_rollout_ref.actor.kl_loss_coef |
|-------|------------------------------|--------------------------------------|
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Qwen/Qwen3-1.7B | 0.01 |
| [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Qwen/Qwen3-4B-Instruct-2507 | 0.1 |

## References

- [Tau-2 Benchmark](https://github.com/sierra-research/tau2-bench)
- [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)