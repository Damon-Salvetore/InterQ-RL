#!/usr/bin/env bash
# InterQ-RL: W8A8-KV8 rollout, W/A/KV QAT learner, execution-prefix correction.
set -euo pipefail
METHOD=interq_rl ROLLOUT=fp8 QAT=1 bash "$(dirname "$0")/common.sh" actor_rollout_ref.actor.policy_loss.loss_mode=interq_rl "$@"
