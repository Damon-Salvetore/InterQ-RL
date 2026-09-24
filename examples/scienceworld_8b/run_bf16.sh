#!/usr/bin/env bash
# BF16 rollout and learner.
set -euo pipefail
METHOD=bf16 ROLLOUT=bf16 bash "$(dirname "$0")/common.sh" "$@"
