#!/usr/bin/env bash
# QuRL (ACR + UAQ) on the W8A8-KV8 rollout; MODEL_PATH is the output of scripts/apply_uaq.py.
set -euo pipefail
METHOD=qurl ROLLOUT=fp8 MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the UAQ-rescaled checkpoint} bash "$(dirname "$0")/common.sh" actor_rollout_ref.actor.policy_loss.loss_mode=qurl "$@"
