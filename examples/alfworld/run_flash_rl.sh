#!/usr/bin/env bash
# Flash-RL (token-local TIS, C = 2) on the W8A8-KV8 rollout.
set -euo pipefail
METHOD=flash_rl ROLLOUT=fp8 bash "$(dirname "$0")/common.sh" algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=2.0 "$@"
