# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""QAT FakeQuantized Linear module for NVFP4 (W4A4/W4A16) with FSDP compatibility.

Includes Triton kernels for high-performance FP4 quantization.
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QATLinear", "QATMode"]


import triton
import triton.language as tl

_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
FP4_E2M1_MAX: float = 6.0
FP8_E4M3_MAX: float = 448.0


# ── FP8 W8A8-dynamic fake-quant (matches vLLM rollout.quantization=fp8) ──────────────
_FP8_WEIGHT_BLOCK = (128, 128)  # overridable from the model's quant_config.weight_block_size
_FP8_ACT_GROUP = 128
_FP8_EPS = 1e-10
_BACKWARD_OPERAND_POLICIES = frozenset({"save", "recompute"})


def _resolve_backward_operand_policy(policy: str) -> str:
    resolved = str(policy).lower()
    if resolved not in _BACKWARD_OPERAND_POLICIES:
        expected = " | ".join(sorted(_BACKWARD_OPERAND_POLICIES))
        raise ValueError(
            f"Unknown backward_operand_policy {policy!r}; expected {expected}"
        )
    return resolved


def _fp8_env_use_ue8m0() -> bool:
    """Read VLLM_USE_DEEP_GEMM_E8M0 (vLLM default "1")."""
    import os

    return bool(int(os.getenv("VLLM_USE_DEEP_GEMM_E8M0", "1")))


def _resolve_fp8_scale_ue8m0(scale_format: str) -> bool:
    """Map ``scale_format`` ("fp32" / "e8m0" / "auto" = VLLM_USE_DEEP_GEMM_E8M0) to the use_ue8m0 flag."""
    fmt = str(scale_format).lower()
    if fmt == "fp32":
        return False
    if fmt == "e8m0":
        return True
    if fmt == "auto":
        return _fp8_env_use_ue8m0()
    raise ValueError(f"Unknown fp8 scale_format {scale_format!r}; expected 'fp32' | 'e8m0' | 'auto'")


_FP8_WEIGHT_EPS_FP32 = 1e-10  # verl scaled_fp8_blockwise floor (fp32 path)
_FP8_WEIGHT_EPS_UE8M0 = 1e-4  # vLLM per_block_cast_to_fp8 amax.clamp(1e-4) (e8m0 path, DeepGEMM)


def _resolve_fp8_scale_round(scale_round: str) -> tuple[bool, bool]:
    """Map ``scale_round`` to (weight_stochastic, act_stochastic).

    "ceil" -> (False, False); "stochastic_weight" -> (True, False);
    "stochastic" -> (True, True).
    """
    r = str(scale_round).lower()
    if r == "ceil":
        return False, False
    if r == "stochastic_weight":
        return True, False
    if r == "stochastic":
        return True, True
    raise ValueError(
        f"Unknown fp8 scale_round {scale_round!r}; expected 'ceil' | 'stochastic_weight' | 'stochastic'"
    )


def _ue8m0_scale(scale: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Round a continuous FP8 scale to a power of two.

    ``stochastic=False`` gives ``2^ceil(log2 s)`` as vLLM/DeepGEMM does; ``stochastic=True`` rounds the
    exponent up with probability equal to its fractional part, so ``E[log2 scale] == log2 s``.
    """
    log_s = torch.log2(scale)
    if not stochastic:
        return torch.exp2(torch.ceil(log_s))
    floor_s = torch.floor(log_s)
    frac = log_s - floor_s
    # Bernoulli(frac): round up with probability == distance to the next power of two.
    up = (torch.rand_like(frac) < frac).to(frac.dtype)
    return torch.exp2(floor_s + up)


@triton.jit
def _fp8_blockwise_dequant_kernel(
    fp8_ptr,
    scale_ptr,
    output_ptr,
    stride_fp8_m,
    stride_fp8_n,
    stride_scale_m,
    stride_scale_n,
    stride_output_m,
    stride_output_n,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.cast(tl.program_id(axis=0), tl.int64)
    block_n = tl.cast(tl.program_id(axis=1), tl.int64)
    offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)

    fp8_values = tl.load(
        fp8_ptr
        + offsets_m[:, None] * stride_fp8_m
        + offsets_n[None, :] * stride_fp8_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr
        + block_m * stride_scale_m
        + block_n * stride_scale_n
    ).to(tl.float32)
    tl.store(
        output_ptr
        + offsets_m[:, None] * stride_output_m
        + offsets_n[None, :] * stride_output_n,
        fp8_values * scale,
        mask=mask,
    )


def _dequantize_fp8_blockwise(
    fp8_data: torch.Tensor,
    descale: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused block-scale FP8 dequantization."""
    if fp8_data.ndim != 2 or descale.ndim != 2:
        raise ValueError("blockwise FP8 dequantization requires 2D data and scales")
    if not fp8_data.is_cuda or not descale.is_cuda:
        raise ValueError("blockwise FP8 dequantization requires CUDA tensors")

    block_m, block_n = block_size
    rows, columns = fp8_data.shape
    expected_scale_shape = (
        triton.cdiv(rows, block_m),
        triton.cdiv(columns, block_n),
    )
    if tuple(descale.shape) != expected_scale_shape:
        raise ValueError(
            f"expected scale shape {expected_scale_shape}, got {tuple(descale.shape)}"
        )

    output = torch.empty_like(fp8_data, dtype=output_dtype)
    grid = (
        triton.cdiv(rows, block_m),
        triton.cdiv(columns, block_n),
    )
    _fp8_blockwise_dequant_kernel[grid](
        fp8_data,
        descale,
        output,
        *fp8_data.stride(),
        *descale.stride(),
        *output.stride(),
        rows,
        columns,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )
    return output


@triton.jit
def _fp8_per_token_group_qdq_kernel(
    input_ptr,
    output_ptr,
    stride_input_t,
    stride_input_h,
    stride_output_t,
    stride_output_h,
    H,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    token = tl.cast(tl.program_id(axis=0), tl.int64)
    group = tl.cast(tl.program_id(axis=1), tl.int64)
    offsets_h = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offsets_h < H

    values = tl.load(
        input_ptr + token * stride_input_t + offsets_h * stride_input_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values)), EPS)
    scale = absmax * (1.0 / FP8_MAX)
    quantized = tl.clamp(
        tl.math.div_rn(values, scale),
        -FP8_MAX,
        FP8_MAX,
    ).to(tl.float8e4nv)
    dequantized = quantized.to(tl.float32) * scale
    tl.store(
        output_ptr
        + token * stride_output_t
        + offsets_h * stride_output_h,
        dequantized,
        mask=mask,
    )


def _fake_quantize_fp8_per_token_group(
    activation: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Fused per-token-group FP8 QDQ."""
    if activation.ndim != 2 or not activation.is_cuda:
        raise ValueError(
            "per-token-group FP8 QDQ requires a 2D CUDA tensor"
        )
    tokens, hidden_size = activation.shape
    if hidden_size % group_size != 0:
        raise ValueError(
            f"hidden {hidden_size} is not divisible by group {group_size}"
        )

    output = torch.empty_like(activation)
    grid = (tokens, hidden_size // group_size)
    _fp8_per_token_group_qdq_kernel[grid](
        activation,
        output,
        *activation.stride(),
        *output.stride(),
        hidden_size,
        EPS=_FP8_EPS,
        FP8_MAX=FP8_E4M3_MAX,
        GROUP_SIZE=group_size,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.jit
def _fp8_per_token_head_qdq_kernel(
    input_ptr,
    output_ptr,
    stride_input_t,
    stride_input_h,
    stride_output_t,
    stride_output_h,
    H,
    FP8_MAX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token = tl.cast(tl.program_id(axis=0), tl.int64)
    head = tl.cast(tl.program_id(axis=1), tl.int64)
    offsets_h = head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    mask = offsets_h < H

    values = tl.load(
        input_ptr + token * stride_input_t + offsets_h * stride_input_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.abs(values))
    scale = tl.maximum(absmax / FP8_MAX, 1e-6)
    quantized = tl.clamp(
        values * tl.math.div_rn(tl.full((), 1.0, tl.float32), scale),
        -FP8_MAX,
        FP8_MAX,
    ).to(tl.float8e4nv)
    dequantized = quantized.to(tl.float32) * scale
    tl.store(
        output_ptr
        + token * stride_output_t
        + offsets_h * stride_output_h,
        dequantized,
        mask=mask,
    )


def _fake_quantize_fp8_per_token_head(
    kv: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Fused per-token-head KV-cache FP8 QDQ."""
    if kv.ndim != 2 or not kv.is_cuda:
        raise ValueError("per-token-head FP8 QDQ requires a 2D CUDA tensor")
    tokens, hidden_size = kv.shape
    if hidden_size % head_dim != 0:
        raise ValueError(
            f"hidden {hidden_size} is not divisible by head_dim {head_dim}"
        )

    output = torch.empty_like(kv)
    grid = (tokens, hidden_size // head_dim)
    _fp8_per_token_head_qdq_kernel[grid](
        kv,
        output,
        *kv.stride(),
        *output.stride(),
        hidden_size,
        FP8_MAX=FP8_E4M3_MAX,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return output



def fp8_fake_quant_weight_blockwise(
    weight: torch.Tensor, block_size=(128, 128), use_ue8m0: bool = False, stochastic: bool = False
) -> torch.Tensor:
    """Blockwise FP8 weight QDQ matching the rollout.

    fp32 scales reuse the rollout's ``scaled_fp8_blockwise`` kernel; E8M0 scales mirror vLLM's
    ``per_block_cast_to_fp8(use_ue8m0=True)``. The caller applies the STE backward.
    """
    orig_shape = weight.shape
    w2d = weight.reshape(-1, orig_shape[-1])
    M, N = w2d.shape
    bm, bn = int(block_size[0]), int(block_size[1])

    if not use_ue8m0:
        from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

        fp8_data, descale = scaled_fp8_blockwise(w2d.to(torch.bfloat16), [bm, bn])
        return _dequantize_fp8_blockwise(
            fp8_data,
            descale,
            (bm, bn),
            weight.dtype,
        ).reshape(orig_shape)

    # vLLM re-casts the synced FP8 weight to E8M0; both roundings are applied.
    from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

    fp8_data, descale = scaled_fp8_blockwise(w2d.to(torch.bfloat16), [bm, bn])
    desc_full = descale.repeat_interleave(bm, dim=0).repeat_interleave(bn, dim=1)[:M, :N]
    w_step1 = (fp8_data.to(torch.float32) * desc_full.to(torch.float32))  # rollout's dequantized weight

    # step 2: per_block_cast_to_fp8(use_ue8m0=True) on the round-tripped weight
    pad_m = (bm - M % bm) % bm
    pad_n = (bn - N % bn) % bn
    wf = torch.zeros((M + pad_m, N + pad_n), dtype=torch.float32, device=w2d.device)
    wf[:M, :N] = w_step1
    Mp, Np = wf.shape
    wv = wf.view(Mp // bm, bm, Np // bn, bn)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(_FP8_WEIGHT_EPS_UE8M0)  # vLLM uses clamp(1e-4)
    scale = amax * (1.0 / FP8_E4M3_MAX)
    scale = _ue8m0_scale(scale, stochastic=stochastic)  # UE8M0 (ceil, or unbiased stochastic rounding)
    w_q = torch.clamp(wv * (1.0 / scale), -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    w_dq = w_q.to(torch.float32) * scale
    w_dq = w_dq.view(Mp, Np)[:M, :N]
    return w_dq.to(weight.dtype).reshape(orig_shape)


def fp8_fake_quant_act_per_token_group(
    x: torch.Tensor, group_size: int = 128, use_ue8m0: bool = False, stochastic: bool = False
) -> torch.Tensor:
    """Per-token-group activation QDQ following vLLM's ``_per_token_group_quant_fp8``.

    scale = max(absmax, eps) * (1 / 448), rounded to a power of two iff use_ue8m0; y_q = clamp(y / scale).
    ``use_ue8m0`` must match the rollout's VLLM_USE_DEEP_GEMM_E8M0.
    """
    orig_shape = x.shape
    H = orig_shape[-1]
    assert H % group_size == 0, f"hidden {H} not divisible by act group {group_size}"
    x2d = x.reshape(-1, H)
    T = x2d.shape[0]
    if not use_ue8m0:
        return _fake_quantize_fp8_per_token_group(
            x2d,
            group_size,
        ).reshape(orig_shape)

    xg = x2d.to(torch.float32).reshape(T, H // group_size, group_size)
    absmax = torch.clamp(xg.abs().amax(dim=-1, keepdim=True), min=_FP8_EPS)
    scale = absmax * (1.0 / FP8_E4M3_MAX)
    if use_ue8m0:
        scale = _ue8m0_scale(scale, stochastic=stochastic)
    x_q = torch.clamp(xg / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    x_dq = x_q.to(torch.float32) * scale
    return x_dq.reshape(orig_shape).to(x.dtype)


class STEFP8QuantWeight(torch.autograd.Function):
    """Straight-Through Estimator for blockwise FP8 weight fake-quant (identity backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size, use_ue8m0: bool = False, stochastic: bool = False):
        return fp8_fake_quant_weight_blockwise(x, block_size, use_ue8m0=use_ue8m0, stochastic=stochastic)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None, None


class STEFP8QuantAct(torch.autograd.Function):
    """Straight-Through Estimator for per-token-group FP8 activation fake-quant (identity backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int, use_ue8m0: bool = False, stochastic: bool = False):
        return fp8_fake_quant_act_per_token_group(x, group_size, use_ue8m0=use_ue8m0, stochastic=stochastic)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None, None


class _FP8QATLinearRematerialize(torch.autograd.Function):
    """FP8 QAT linear that rematerializes deterministic QDQ operands in backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        weight_block: tuple[int, int],
        act_group: int,
        use_ue8m0: bool,
        weight_stochastic: bool,
        act_stochastic: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.weight_block = weight_block
        ctx.act_group = act_group
        ctx.use_ue8m0 = use_ue8m0
        ctx.weight_stochastic = weight_stochastic
        ctx.act_stochastic = act_stochastic

        weight_fq = fp8_fake_quant_weight_blockwise(
            weight,
            weight_block,
            use_ue8m0=use_ue8m0,
            stochastic=weight_stochastic,
        )
        x_fq = fp8_fake_quant_act_per_token_group(
            x,
            act_group,
            use_ue8m0=use_ue8m0,
            stochastic=act_stochastic,
        )
        return F.linear(x_fq, weight_fq, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        x, weight = ctx.saved_tensors
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_input = None
        if ctx.needs_input_grad[0]:
            weight_fq = fp8_fake_quant_weight_blockwise(
                weight,
                ctx.weight_block,
                use_ue8m0=ctx.use_ue8m0,
                stochastic=ctx.weight_stochastic,
            )
            grad_input = grad_output_2d.matmul(weight_fq).reshape_as(x)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            x_fq = fp8_fake_quant_act_per_token_group(
                x,
                ctx.act_group,
                use_ue8m0=ctx.use_ue8m0,
                stochastic=ctx.act_stochastic,
            )
            grad_weight = grad_output_2d.transpose(0, 1).matmul(
                x_fq.reshape(-1, x_fq.shape[-1])
            )

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


def fp8_fake_quant_kv_per_token_head(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Per-(token, head) KV QDQ, op for op as vLLM's ``_reshape_cache_per_token_head``.

    scale = max(absmax / 448, 1e-6) over one head_dim slice; q = clamp(x * (1 / scale)).
    ``x`` is shaped [..., num_heads * head_dim].
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if x2d.is_cuda:
        return _fake_quantize_fp8_per_token_head(
            x2d,
            head_dim,
        ).reshape(orig_shape)

    xh = x.reshape(*orig_shape[:-1], orig_shape[-1] // head_dim, head_dim).to(torch.float32)
    scale = torch.clamp(xh.abs().amax(dim=-1, keepdim=True) / FP8_E4M3_MAX, min=1e-6)
    x_q = torch.clamp(xh * (1.0 / scale), -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    x_dq = x_q.to(torch.float32) * scale
    return x_dq.reshape(orig_shape).to(x.dtype)


class STEFP8QuantKV(torch.autograd.Function):
    """Straight-Through Estimator for per-token-head FP8 KV fake-quant (identity backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, head_dim: int):
        return fp8_fake_quant_kv_per_token_head(x, head_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None


@triton.jit
def _fp4_fake_quant_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    FP4_MAX: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)

    block_max = tl.max(x_abs, axis=2, keep_dims=True)
    block_max_scaled = block_max / (FP4_MAX * global_scale_safe)
    block_max_scaled = tl.minimum(block_max_scaled, FP8_MAX)
    block_max_quant = block_max_scaled.to(tl.float8e4nv).to(tl.float32) * global_scale
    block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)

    block_max_quant_broadcast = tl.broadcast_to(block_max_quant, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    abs_scaled = x_abs / block_max_quant_broadcast

    q_val = tl.where(
        abs_scaled <= 0.25,
        0.0,
        tl.where(
            abs_scaled < 0.75,
            0.5,
            tl.where(
                abs_scaled <= 1.25,
                1.0,
                tl.where(
                    abs_scaled < 1.75,
                    1.5,
                    tl.where(
                        abs_scaled <= 2.5,
                        2.0,
                        tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, FP4_MAX)),
                    ),
                ),
            ),
        ),
    )

    x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = tl.where(tile_reshaped >= 0, x_rescaled, -x_rescaled)
    tile_quant = tl.reshape(x_rescaled, (TILE_M, TILE_N))

    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))


def fp4_fake_quant_weight(
    weight: torch.Tensor,
    global_amax: torch.Tensor = None,
    block_size: int = 16,
    tile_rows: int = 16,
    tile_cols: int = 64,
) -> torch.Tensor:
    """Apply FP4 fake quantization using Triton kernel."""
    x_shape = weight.shape
    x_dtype = weight.dtype
    x = weight.reshape(-1, x_shape[-1]).contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    if global_amax is None:
        global_amax = weight.abs().max().to(torch.float32)
    global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

    grid = (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    _fp4_fake_quant_kernel[grid](
        x,
        y,
        M,
        N,
        global_scale,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_SIZE=block_size,
        TILE_M=tile_rows,
        TILE_N=tile_cols_aligned,
        NUM_FP4_BLOCKS=num_fp4_blocks,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[x_dtype],
        FP4_MAX=FP4_E2M1_MAX,
        FP8_MAX=FP8_E4M3_MAX,
    )
    return y.view(*x_shape)


class STEFP4QuantTriton(torch.autograd.Function):
    """Straight-Through Estimator wrapper for Triton FP4 quantization kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, global_amax: torch.Tensor, block_size: int) -> torch.Tensor:
        return fp4_fake_quant_weight(x, global_amax=global_amax, block_size=block_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None


class QATMode(str, Enum):
    """QAT quantization mode."""

    W4A4 = "w4a4"  # Weight 4-bit, Activation 4-bit (dynamic)
    W4A16 = "w4a16"  # Weight 4-bit, Activation 16-bit (weight only)
    W8A8_FP8 = "w8a8_fp8"  # e4m3 weights (128x128 blocks) and dynamic e4m3 activations (1x128 groups)
    W8A16_FP8 = "w8a16_fp8"  # e4m3 weights only


class QATLinear(nn.Linear):
    """QAT FakeQuantized Linear layer with FSDP compatibility."""

    _UNINITIALIZED_SCALE = -1.0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",  # Observer strategy for activation global_scale
        scale_format: str = "auto",  # fp8 W8A8 scale format: "fp32" | "e8m0" | "auto" (follow env)
        scale_round: str = "ceil",  # e8m0 scale rounding: "ceil" | "stochastic_weight" | "stochastic"
        backward_operand_policy: str = "save",  # "save" | "recompute" deterministic QDQ operands in backward
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer

        self._weight_blockwise_scale: Optional[torch.Tensor] = None
        self._weight_global_scale: Optional[torch.Tensor] = None
        self._cached_weight_amax: Optional[torch.Tensor] = None
        self._fusion_siblings_ref = None

        # FP8 W8A8 layout (block sizes; may be overridden from the model quant_config in apply_qat)
        self._fp8_weight_block = _FP8_WEIGHT_BLOCK
        self._fp8_act_group = _FP8_ACT_GROUP
        self.scale_format = scale_format
        self._fp8_use_ue8m0 = _resolve_fp8_scale_ue8m0(scale_format)
        self._fp8_act_use_ue8m0 = self._fp8_use_ue8m0
        self.scale_round = scale_round
        self._fp8_scale_stochastic, self._fp8_act_scale_stochastic = _resolve_fp8_scale_round(scale_round)
        self.backward_operand_policy = _resolve_backward_operand_policy(
            backward_operand_policy
        )
        if (
            self.backward_operand_policy == "recompute"
            and self.mode != QATMode.W8A8_FP8
        ):
            raise ValueError(
                "backward_operand_policy='recompute' currently supports only "
                "mode='w8a8_fp8'"
            )
        if (
            self.backward_operand_policy == "recompute"
            and self._fp8_use_ue8m0
            and (
                self._fp8_scale_stochastic
                or self._fp8_act_scale_stochastic
            )
        ):
            raise ValueError(
                "backward_operand_policy='recompute' requires deterministic "
                "QDQ; stochastic e8m0 scale rounding is unsupported"
            )

        if mode == QATMode.W4A4:
            self.register_buffer(
                "input_global_scale", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )

            self.register_buffer(
                "input_amax", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )

            self._ema_decay: float = 0.01

        self.fake_quant_enabled = True

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",
        scale_format: str = "auto",
        scale_round: str = "ceil",
        backward_operand_policy: str = "save",
    ) -> "QATLinear":
        """Create QATLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None

        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            mode=mode,
            group_size=group_size,
            activation_observer=activation_observer,
            scale_format=scale_format,
            scale_round=scale_round,
            backward_operand_policy=backward_operand_policy,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def _is_amax_initialized(self) -> bool:
        """Check if input_amax has been initialized."""
        if not hasattr(self, "input_amax"):
            return False
        return self.input_amax.item() != self._UNINITIALIZED_SCALE

    def _update_input_global_scale(self, x: torch.Tensor):
        """Update static input_global_scale based on observer strategy."""
        assert self.mode == QATMode.W4A4, "_update_input_global_scale should only be called in W4A4 mode"

        current_amax = torch.amax(torch.abs(x)).detach().to(torch.float32)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(current_amax, op=torch.distributed.ReduceOp.MAX)

        scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX

        if self.activation_observer == "memoryless_minmax":
            new_scale = (scale_factor / (current_amax + 1e-12)).view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "static_minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = torch.maximum(self.input_amax, current_amax.view(1).to(self.input_amax.device))
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = (1 - self._ema_decay) * self.input_amax + self._ema_decay * current_amax.view(1).to(
                    self.input_amax.device
                )
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        else:
            raise ValueError(f"Unknown activation_observer: {self.activation_observer}")

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to weight tensor using Triton kernel."""
        with torch.no_grad():
            if self._cached_weight_amax is not None:
                global_amax = self._cached_weight_amax
            else:
                siblings_ref = getattr(self, "_fusion_siblings_ref", None)

                if siblings_ref is not None:
                    siblings = [ref() for ref in siblings_ref if ref() is not None]
                    siblings = [s for s in siblings if s.weight.device != torch.device("meta")]

                    for sibling in siblings:
                        sibling_amax = getattr(sibling, "_cached_weight_amax", None)
                        if sibling_amax is not None:
                            global_amax = sibling_amax
                            self._cached_weight_amax = global_amax
                            break
                    else:
                        all_modules = [self] + siblings
                        amaxes = [m.weight.abs().max().to(torch.float32) for m in all_modules]
                        global_amax = torch.max(torch.stack(amaxes))

                        self._cached_weight_amax = global_amax
                        for sibling in siblings:
                            sibling._cached_weight_amax = global_amax
                else:
                    global_amax = weight.abs().max().to(torch.float32)
                    self._cached_weight_amax = global_amax

            if self._weight_global_scale is None:
                self._weight_global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

        result = STEFP4QuantTriton.apply(weight, global_amax, self.group_size)

        return result

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to activation tensor (W4A4 mode only)."""
        original_shape = x.shape

        if x.dim() == 3:
            x_2d = x.view(-1, x.shape[-1])
        else:
            x_2d = x

        if self.training:
            self._update_input_global_scale(x_2d)

        if self.input_global_scale.item() == self._UNINITIALIZED_SCALE:
            raise RuntimeError("W4A4 input_global_scale uninitialized. Load PTQ model first.")

        global_amax = (FP4_E2M1_MAX * FP8_E4M3_MAX) / self.input_global_scale.to(x.device)
        result = STEFP4QuantTriton.apply(x_2d, global_amax, self.group_size)
        return result.view(original_shape)

    def _fake_quantize_weight_fp8(self, weight: torch.Tensor) -> torch.Tensor:
        """FP8 W8A8 weight fake-quant (blockwise e4m3); scale recomputed every forward like vLLM. STE identity grad.

        use_ue8m0 tracks the rollout's scale format (fp32 vs power-of-two); see _resolve_fp8_scale_ue8m0.
        """
        return STEFP8QuantWeight.apply(
            weight, self._fp8_weight_block, self._fp8_use_ue8m0, self._fp8_scale_stochastic
        )

    def _fake_quantize_activation_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """FP8 W8A8 activation fake-quant (per-token-group dynamic e4m3). STE identity grad.

        use_ue8m0 tracks the rollout's scale format (fp32 vs power-of-two); see _resolve_fp8_scale_ue8m0.
        """
        return STEFP8QuantAct.apply(x, self._fp8_act_group, self._fp8_act_use_ue8m0, self._fp8_act_scale_stochastic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization."""
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        if self.mode == QATMode.W8A8_FP8:
            if self.backward_operand_policy == "recompute":
                return _FP8QATLinearRematerialize.apply(
                    x,
                    self.weight,
                    self.bias,
                    self._fp8_weight_block,
                    self._fp8_act_group,
                    self._fp8_use_ue8m0,
                    self._fp8_scale_stochastic,
                    self._fp8_act_scale_stochastic,
                )
            weight_fq = self._fake_quantize_weight_fp8(self.weight)
            x_fq = self._fake_quantize_activation_fp8(x)
            return F.linear(x_fq, weight_fq, self.bias)

        if self.mode == QATMode.W8A16_FP8:
            weight_fq = self._fake_quantize_weight_fp8(self.weight)
            return F.linear(x, weight_fq, self.bias)

        weight_fq = self._fake_quantize_weight(self.weight)

        if self.mode == QATMode.W4A4:
            x_fq = self._fake_quantize_activation(x)
        else:
            x_fq = x

        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        extra = ""
        if self.mode in (QATMode.W8A8_FP8, QATMode.W8A16_FP8):
            extra = (
                f", scale_format={self.scale_format}, use_ue8m0={self._fp8_use_ue8m0}"
                f", scale_round={self.scale_round}"
                f", backward_operand_policy={self.backward_operand_policy}"
            )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, fake_quant_enabled={self.fake_quant_enabled}{extra}"
        )
