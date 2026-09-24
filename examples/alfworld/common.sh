#!/usr/bin/env bash
# ALFWorld / Qwen3-4B-Instruct-2507 on one node of 8 GPUs.
# Method scripts set METHOD, ROLLOUT (bf16 | fp8) and QAT (0 | 1), and append their own overrides.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
DATA_DIR=${ALFWORLD_DATA_ROOT:-data/alfworld}
METHOD=${METHOD:?}
ROLLOUT=${ROLLOUT:-bf16}
QAT=${QAT:-0}

export ALFWORLD_DATA_ROOT=$DATA_DIR
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0

QUANT_ARGS=()
if [[ $ROLLOUT == fp8 ]]; then
    QUANT_ARGS+=(
        actor_rollout_ref.rollout.quantization=fp8
        +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8_per_token_head
    )
fi
if [[ $QAT == 1 ]]; then
    QUANT_ARGS+=(
        actor_rollout_ref.actor.fsdp_config.qat.enable=true
        actor_rollout_ref.actor.fsdp_config.qat.mode=w8a8_fp8
        actor_rollout_ref.actor.fsdp_config.qat.scale_format=fp32
        actor_rollout_ref.actor.fsdp_config.qat.quantization_config_path=verl/utils/qat/fp8_w8a8_dynamic.json
        actor_rollout_ref.actor.fsdp_config.qat.quantize_kv_cache=true
    )
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    critic.enable=false \
    reward_model.enable=false \
    data.train_files="['$DATA_DIR/train.parquet']" \
    data.val_files="['$DATA_DIR/valid_seen.parquet','$DATA_DIR/valid_unseen.parquet']" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=20480 \
    actor_rollout_ref.rollout.max_model_len=24576 \
    actor_rollout_ref.rollout.max_num_batched_tokens=24576 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=agent_system/environments/alfworld/config/alfworld_agent.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_agent \
    ++trainer.worker_env.ALFWORLD_DATA_ROOT=$DATA_DIR \
    trainer.logger='["console"]' \
    trainer.project_name=interq_rl \
    trainer.experiment_name=alfworld-qwen3_4b-$METHOD \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=400 \
    trainer.test_freq=10 \
    trainer.save_freq=100 \
    trainer.log_val_generations=10 \
    "${QUANT_ARGS[@]}" \
    "$@"
