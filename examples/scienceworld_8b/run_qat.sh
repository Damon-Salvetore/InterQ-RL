#!/usr/bin/env bash
# QAT: W8A8-KV8 rollout with the W/A/KV QAT learner, no correction.
set -euo pipefail
METHOD=qat ROLLOUT=fp8 QAT=1 bash "$(dirname "$0")/common.sh" "$@"
