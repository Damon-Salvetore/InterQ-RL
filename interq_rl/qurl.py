# Copyright 2026 The InterQ-RL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QuRL Adaptive Clipping Range (ACR) weights, consumed by the `qurl` loss."""

from __future__ import annotations

import torch

__all__ = ["compute_acr_weights"]


def compute_acr_weights(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    cap: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scale = min(rho, C) and r = min(1, C / rho) for rho = pi_old / pi_rollout; the upper bound is (1 + eps) / r."""

    if old_log_prob.shape != rollout_log_prob.shape:
        raise ValueError("old and rollout log probabilities must share shape")
    if old_log_prob.shape != response_mask.shape:
        raise ValueError("log probabilities and response_mask must share shape")
    if not (cap > 1.0):
        raise ValueError(f"ACR cap must exceed 1.0, got {cap}")

    log_ratio = torch.clamp(old_log_prob - rollout_log_prob, min=-20.0, max=20.0)
    raw = torch.exp(log_ratio)
    scale = torch.clamp(raw, max=cap) * response_mask
    clip_r = torch.clamp(cap / torch.clamp(raw, min=1e-6), max=1.0) * response_mask

    return scale.detach(), clip_r.detach()
