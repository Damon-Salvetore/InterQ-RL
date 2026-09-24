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

"""AIS batch-level mixed importance weights."""

import math

import torch


def _validate_same_shape(**tensors: torch.Tensor) -> None:
    values = list(tensors.values())
    if not values or any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("all AIS inputs must be torch tensors")
    shape = values[0].shape
    device = values[0].device
    if values[0].ndim != 2:
        raise ValueError("AIS tensors must be rank-2 [batch, response_length]")
    if any(value.shape != shape for value in values):
        raise ValueError("all AIS tensors must have the same shape")
    if any(value.device != device for value in values):
        raise ValueError("all AIS tensors must be on the same device")


@torch.no_grad()
def compute_ais_weights(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip: float = 5.0,
    delta: float = 0.02,
    gamma: float = 1.2,
    beta_var: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """AIS token weights over one batch."""

    _validate_same_shape(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        advantages=advantages,
        response_mask=response_mask,
    )
    for name, value in {
        "clip": clip,
        "delta": delta,
        "gamma": gamma,
        "epsilon": epsilon,
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(beta_var) or beta_var < 0:
        raise ValueError("beta_var must be finite and nonnegative")

    valid_mask = response_mask == 1
    for name, tensor in {
        "old_log_prob": old_log_prob,
        "rollout_log_prob": rollout_log_prob,
        "advantages": advantages,
    }.items():
        if not torch.isfinite(tensor[valid_mask]).all().item():
            raise ValueError(f"{name} contains non-finite valid-token values")

    work_dtype = torch.float64 if any(
        tensor.dtype == torch.float64
        for tensor in (old_log_prob, rollout_log_prob, advantages)
    ) else torch.float32
    weights = torch.ones_like(old_log_prob, dtype=work_dtype)
    count = int(valid_mask.sum().item())
    if count == 0:
        return weights

    execution_log_ratio = (
        old_log_prob[valid_mask].to(work_dtype)
        - rollout_log_prob[valid_mask].to(work_dtype)
    )
    truncated = torch.exp(
        torch.minimum(execution_log_ratio, execution_log_ratio.new_tensor(math.log(clip)))
    )
    valid_advantages = advantages[valid_mask].to(work_dtype)

    alpha_ess = truncated.mean() / torch.sqrt(truncated.square().mean())
    alpha_mis = torch.clamp(execution_log_ratio.abs().mean() / delta, max=1.0)
    advantage_std = valid_advantages.std(unbiased=False)
    weighted_advantage_std = (valid_advantages * truncated).std(unbiased=False)
    delta_sigma = weighted_advantage_std / (advantage_std + epsilon)
    alpha_var = torch.clamp_min((delta_sigma - gamma) / gamma, 0.0)
    alpha = torch.clamp(alpha_ess - beta_var * alpha_var, 0.0, 1.0) * alpha_mis

    weights[valid_mask] = 1.0 + alpha * (truncated - 1.0)
    return weights
