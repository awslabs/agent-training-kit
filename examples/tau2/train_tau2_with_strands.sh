#!/bin/bash
# examples/tau2/train_tau2_with_strands.sh

set -x
ulimit -c 0

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export BYPASS_TOOL_CONSENT=true

# Set Tau-2 data directory (adjust path as needed)
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-./data/tau2/domains}"

LOG_DIR=/tmp/logs
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/tau2_training_"$(date +%Y%m%d_%H%M%S)".log"
EXP_NAME="tau2-retail-and-airline"

python3 -m examples.tau2.train_tau2_with_strands \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.val_batch_size=700 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_mini_batch=False \
    actor_rollout_ref.actor.ppo_num_mini_batches=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.clip_ratio_high=0.1 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.grad_norm_threshold=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len=65536 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=False \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='trainable-strands-agent' \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=3 \
    trainer.default_hdfs_dir=null \
    trainer.rejection_sample=False \
    agent.max_steps=18 \
    agent.async_engine=True \
    agent.use_stepwise_advantage=False \
    agent.stepwise_advantage_mode="mc_return" \
    +agent.engine_args.strands_timeout=300 \
    trainer.total_epochs=100 2>&1 | tee "$LOG_FILE"

