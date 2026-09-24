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
"""InterQ-RL: execution-prefix correction with support-aware allocation."""

from __future__ import annotations

import math

import torch

__all__ = ["compute_interq_weights"]

LOG_RATIO_BOUND = 20.0
BISECTION_STEPS = 80


def _check_inputs(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    if old_log_prob.ndim != 2:
        raise ValueError("log probabilities must be rank-2 [batch, response_length]")
    if old_log_prob.shape != rollout_log_prob.shape or old_log_prob.shape != response_mask.shape:
        raise ValueError("old_log_prob, rollout_log_prob and response_mask must share shape")
    if old_log_prob.device != rollout_log_prob.device or old_log_prob.device != response_mask.device:
        raise ValueError("all InterQ-RL tensors must share a device")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("response_mask must be binary")
    valid = response_mask == 1
    if not torch.isfinite(old_log_prob[valid]).all() or not torch.isfinite(rollout_log_prob[valid]).all():
        raise ValueError("log probabilities must be finite on policy tokens")


def compute_prefix_weights(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    eps_low: float,
    eps_high: float,
    exponent: float,
) -> torch.Tensor:
    """W_bar_t = exp(clip(E_t, -eps_low * n_t ** exponent, eps_high * n_t ** exponent)) on policy tokens."""
    valid = response_mask == 1
    mask = valid.to(old_log_prob.dtype)
    log_ratio = torch.clamp(old_log_prob - rollout_log_prob, min=-LOG_RATIO_BOUND, max=LOG_RATIO_BOUND)
    prefix_log_ratio = torch.cumsum(log_ratio * mask, dim=-1)
    envelope = torch.cumsum(mask, dim=-1).clamp(min=1.0).pow(exponent)
    clipped = torch.clamp(prefix_log_ratio, -eps_low * envelope, eps_high * envelope)
    return torch.where(valid, torch.exp(clipped), torch.zeros_like(clipped))


def index_action_segments(response_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-global segment id per token (-1 off-policy) and the length of each action segment."""
    valid = response_mask == 1
    previous = torch.zeros_like(valid)
    if valid.shape[1] > 1:
        previous[:, 1:] = valid[:, :-1]
    starts = valid & ~previous
    local_segment = torch.cumsum(starts.to(torch.long), dim=-1) - 1
    segments_per_row = starts.sum(dim=-1, dtype=torch.long)
    row_offsets = torch.cumsum(segments_per_row, dim=0) - segments_per_row
    token_segment = torch.where(
        valid,
        local_segment + row_offsets[:, None],
        torch.full_like(local_segment, -1),
    )
    num_segments = int(segments_per_row.sum().item())
    if num_segments == 0:
        return token_segment, torch.zeros(0, dtype=torch.long, device=response_mask.device)
    segment_lengths = torch.bincount(token_segment[valid], minlength=num_segments)
    return token_segment, segment_lengths


def _segment_statistics(
    prefix_weights: torch.Tensor,
    token_segment: torch.Tensor,
    segment_lengths: torch.Tensor,
    eps_mom: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """mu_s, nu_s, rho_s and C_s; rho_s = (ESS_s - 1) / (n_s - 1) * nu_s / (nu_s + eps_mom), clipped to [0, 1]."""
    valid = token_segment >= 0
    segment = token_segment[valid]
    weights = prefix_weights[valid]
    num_segments = segment_lengths.numel()
    sum_weights = prefix_weights.new_zeros(num_segments).index_add_(0, segment, weights)
    sum_squares = prefix_weights.new_zeros(num_segments).index_add_(0, segment, weights.square())
    count = segment_lengths.to(prefix_weights.dtype)
    mu = sum_weights / count
    nu = sum_squares / count

    centered = (sum_squares - sum_weights.square() / count).clamp_min(0.0)
    variance = torch.where(
        segment_lengths > 1,
        centered / (count - 1.0).clamp_min(1.0),
        torch.zeros_like(centered),
    )
    signal = torch.clamp_min(mu.square() - variance / count, 0.0)
    support = (signal / (nu + eps_mom)).clamp(0.0, 1.0)
    multi_token = segment_lengths > 1
    fallback = support[multi_token].mean() if multi_token.any() else support.new_ones(())
    support = torch.where(multi_token, support, fallback)

    displacement = (nu - 2.0 * mu + 1.0).clamp_min(0.0)
    return mu, nu, support, displacement


@torch.no_grad()
def allocate_retention(
    support: torch.Tensor,
    displacement: torch.Tensor,
    alpha_g: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """alpha_s = min(c * rho_s, 1), with c found by bisection so that sum C_s alpha_s^2 = alpha_g^2 sum C_s."""
    if support.numel() == 0:
        return support.clone()
    if float(alpha_g.item()) == 0.0:
        return torch.zeros_like(support)
    if float(alpha_g.item()) == 1.0:
        return torch.ones_like(support)
    total_displacement = displacement.sum()
    if float(total_displacement.item()) <= eps:
        return torch.full_like(support, alpha_g)

    score = support.clamp_min(eps)
    budget = alpha_g.square() * total_displacement
    lower = score.new_zeros(())
    upper = score.reciprocal().max()
    for _ in range(BISECTION_STEPS):
        middle = (lower + upper) / 2.0
        used = (displacement * torch.clamp(middle * score, max=1.0).square()).sum()
        lower = torch.where(used < budget, middle, lower)
        upper = torch.where(used < budget, upper, middle)
    return torch.clamp(((lower + upper) / 2.0) * score, max=1.0)


@torch.no_grad()
def compute_interq_weights(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    eps_low: float = 0.025,
    eps_high: float = 0.05,
    exponent: float = 0.5,
    eps_mom: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InterQ-RL coefficients W_t = 1 + alpha_seg(t) * (W_bar_t - 1), with metrics."""
    for name, value in {"eps_low": eps_low, "eps_high": eps_high, "eps_mom": eps_mom}.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(exponent):
        raise ValueError("exponent must be finite")
    _check_inputs(old_log_prob, rollout_log_prob, response_mask)

    prefix_weights = compute_prefix_weights(
        old_log_prob,
        rollout_log_prob,
        response_mask,
        eps_low=eps_low,
        eps_high=eps_high,
        exponent=exponent,
    )
    token_segment, segment_lengths = index_action_segments(response_mask)
    valid = token_segment >= 0
    weights = torch.ones_like(prefix_weights)
    metrics: dict[str, float] = {"interq_rl/segment/count": float(segment_lengths.numel())}

    if segment_lengths.numel():
        mu, nu, support, displacement = _segment_statistics(prefix_weights, token_segment, segment_lengths, eps_mom)
        alpha_g = (mu.mean() / torch.sqrt(nu.mean() + eps_mom)).clamp(0.0, 1.0)
        alpha = allocate_retention(support, displacement, alpha_g)
        weights[valid] = 1.0 + alpha[token_segment[valid]] * (prefix_weights[valid] - 1.0)

        used = (displacement * alpha.square()).sum()
        budget = alpha_g.square() * displacement.sum()
        metrics.update(
            {
                "interq_rl/alpha_g": float(alpha_g.item()),
                "interq_rl/segment/length_mean": float(segment_lengths.to(torch.float64).mean().item()),
                "interq_rl/segment/alpha_mean": float(alpha.mean().item()),
                "interq_rl/segment/alpha_min": float(alpha.min().item()),
                "interq_rl/segment/alpha_max": float(alpha.max().item()),
                "interq_rl/segment/alpha_std": float(alpha.std(unbiased=False).item()),
                "interq_rl/segment/support_mean": float(support.mean().item()),
                "interq_rl/segment/up_fraction": float((alpha > alpha_g + eps_mom).float().mean().item()),
                "interq_rl/segment/down_fraction": float((alpha < alpha_g - eps_mom).float().mean().item()),
                "interq_rl/budget/error": float((used - budget).abs().item()),
                "interq_rl/budget/utilization": float((used / budget).item() if budget > eps_mom else 0.0),
            }
        )

    valid_weights = weights[valid]
    if valid_weights.numel():
        metrics.update(
            {
                "interq_rl/weight/relative_ess": float(
                    (valid_weights.sum().square() / (valid_weights.numel() * valid_weights.square().sum())).item()
                ),
                "interq_rl/weight/mean": float(valid_weights.mean().item()),
                "interq_rl/weight/min": float(valid_weights.min().item()),
                "interq_rl/weight/max": float(valid_weights.max().item()),
                "interq_rl/weight/abs_log_p99": float(torch.quantile(valid_weights.log().abs(), 0.99).item()),
            }
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"non-finite InterQ-RL metrics: {metrics}")
    return weights, metrics
