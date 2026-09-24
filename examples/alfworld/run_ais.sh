#!/usr/bin/env bash
# AIS on the W8A8-KV8 rollout.
set -euo pipefail
METHOD=ais ROLLOUT=fp8 bash "$(dirname "$0")/common.sh" actor_rollout_ref.actor.policy_loss.loss_mode=ais "$@"
