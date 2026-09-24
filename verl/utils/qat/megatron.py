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

"""FP8 fake-quant QAT for Megatron Transformer Engine modules."""

from __future__ import annotations

import contextlib
import fnmatch
import re
import threading
import types
from dataclasses import dataclass
from typing import Any

import torch

from verl.utils.qat.linear import (
    STEFP8QuantAct,
    STEFP8QuantKV,
    STEFP8QuantWeight,
    fp8_fake_quant_act_per_token_group,
    fp8_fake_quant_weight_blockwise,
)

_WEIGHT_BLOCK = (128, 128)
_ACT_GROUP = 128
_PATCH_LOCK = threading.RLock()


def _field(config: Any, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass
class MegatronFP8QATState:
    """Shared on/off state for every fake-quantized module in one model."""

    enabled: bool = True


@contextlib.contextmanager
def _replace_attribute(owner, name: str, replacement):
    """Temporarily replace a TE functional with exception-safe restoration."""
    with _PATCH_LOCK:
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        try:
            yield original
        finally:
            setattr(owner, name, original)


def _quantize_activation_ste(activation: torch.Tensor) -> torch.Tensor:
    if activation.shape[-1] % _ACT_GROUP != 0:
        raise ValueError(
            f"FP8 QAT activation width {activation.shape[-1]} is not divisible by {_ACT_GROUP}"
        )
    return STEFP8QuantAct.apply(activation, _ACT_GROUP, False, False)


def _quantize_weight_ste(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-1] % _WEIGHT_BLOCK[1] != 0:
        raise ValueError(
            f"FP8 QAT weight input width {weight.shape[-1]} is not divisible by {_WEIGHT_BLOCK[1]}"
        )
    return STEFP8QuantWeight.apply(weight, _WEIGHT_BLOCK, False, False)


def _quantize_activation_value(activation: torch.Tensor) -> torch.Tensor:
    return fp8_fake_quant_act_per_token_group(
        activation,
        group_size=_ACT_GROUP,
        use_ue8m0=False,
        stochastic=False,
    )


def _quantize_weight_value(weight: torch.Tensor) -> torch.Tensor:
    return fp8_fake_quant_weight_blockwise(
        weight,
        block_size=_WEIGHT_BLOCK,
        use_ue8m0=False,
        stochastic=False,
    )


def _matches_ignore(name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern == "lm_head" and (
            name == "output_layer" or name.endswith(".output_layer")
        ):
            return True
        if pattern.startswith("re:"):
            if re.match(pattern[3:], name):
                return True
        elif fnmatch.fnmatch(name, pattern) or pattern in name:
            return True
    return False


def resolve_megatron_qat_backend(mode: str) -> str:
    """Map a QAT mode to its Megatron implementation."""
    normalized = str(mode).lower()
    if normalized == "w4a16":
        return "modelopt"
    if normalized == "w8a8_fp8":
        return "verl_fp8"
    raise ValueError(f"Unsupported Megatron QAT mode: {mode!r}")


def _install_linear(module, state: MegatronFP8QATState) -> None:
    import transformer_engine.pytorch.module.linear as te_linear

    original_forward = module.forward

    def forward(_self, *args, **kwargs):
        if not state.enabled:
            return original_forward(*args, **kwargs)

        functional_name = "apply" if torch.is_grad_enabled() else "forward"
        original_functional = getattr(te_linear._Linear, functional_name)
        weight_index = 0 if functional_name == "apply" else 1
        activation_index = weight_index + 2

        def qat_functional(*functional_args, **functional_kwargs):
            prefix = functional_args[:weight_index]
            weight = _quantize_weight_ste(functional_args[weight_index])
            weight_workspace = functional_args[weight_index + 1]
            activation = _quantize_activation_ste(
                functional_args[activation_index]
            )
            suffix = functional_args[activation_index + 1 :]
            return original_functional(
                *prefix,
                weight,
                weight_workspace,
                activation,
                *suffix,
                **functional_kwargs,
            )

        with _replace_attribute(te_linear._Linear, functional_name, qat_functional):
            return original_forward(*args, **kwargs)

    module._verl_fp8_qat_original_forward = original_forward
    module.forward = types.MethodType(forward, module)


def _install_grouped_linear(module, state: MegatronFP8QATState) -> None:
    import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear

    original_forward = module.forward

    def forward(_self, *args, **kwargs):
        if not state.enabled:
            return original_forward(*args, **kwargs)

        functional_name = "apply" if torch.is_grad_enabled() else "forward"
        original_functional = getattr(te_grouped_linear._GroupedLinear, functional_name)
        input_index = 0 if functional_name == "apply" else 1

        def qat_functional(*functional_args, **functional_kwargs):
            num_gemms = module.num_gemms
            activation = _quantize_activation_ste(functional_args[input_index])
            weights_and_biases = functional_args[-2 * num_gemms :]
            weights = [
                _quantize_weight_ste(weight)
                for weight in weights_and_biases[:num_gemms]
            ]
            biases = weights_and_biases[num_gemms:]
            return original_functional(
                *functional_args[:input_index],
                activation,
                *functional_args[input_index + 1 : -2 * num_gemms],
                *weights,
                *biases,
                **functional_kwargs,
            )

        with _replace_attribute(
            te_grouped_linear._GroupedLinear,
            functional_name,
            qat_functional,
        ):
            return original_forward(*args, **kwargs)

    module._verl_fp8_qat_original_forward = original_forward
    module.forward = types.MethodType(forward, module)


def _build_layernorm_linear_function(state: MegatronFP8QATState):
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear

    original_function = te_layernorm_linear._LayerNormLinear

    class QATLayerNormLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, ln_weight, ln_bias, weight, *args, **kwargs):
            quantized_weight = _quantize_weight_value(weight)
            quantized_weight.requires_grad = weight.requires_grad
            original_gemm = te_layernorm_linear.general_gemm

            def qat_gemm(gemm_weight, gemm_input, *gemm_args, **gemm_kwargs):
                quantized_input = _quantize_activation_value(gemm_input)
                if ctx is not None:
                    ctx.verl_qat_input = quantized_input
                return original_gemm(
                    gemm_weight,
                    quantized_input,
                    *gemm_args,
                    **gemm_kwargs,
                )

            with _replace_attribute(
                te_layernorm_linear,
                "general_gemm",
                qat_gemm,
            ):
                return original_function.forward(
                    ctx,
                    inp,
                    ln_weight,
                    ln_bias,
                    quantized_weight,
                    *args,
                    **kwargs,
                )

        @staticmethod
        def backward(ctx, *grad_outputs):
            original_gemm = te_layernorm_linear.general_gemm
            gemm_count = 0

            def qat_gemm(a, b, *gemm_args, **gemm_kwargs):
                nonlocal gemm_count
                gemm_count += 1
                if gemm_count == 2:
                    a = ctx.verl_qat_input
                return original_gemm(a, b, *gemm_args, **gemm_kwargs)

            with _replace_attribute(
                te_layernorm_linear,
                "general_gemm",
                qat_gemm,
            ):
                gradients = original_function.backward(ctx, *grad_outputs)
            del ctx.verl_qat_input
            return gradients

    return QATLayerNormLinear


def _install_layernorm_linear(module, state: MegatronFP8QATState) -> None:
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear

    original_forward = module.forward
    qat_function = _build_layernorm_linear_function(state)

    def forward(_self, *args, **kwargs):
        if not state.enabled:
            return original_forward(*args, **kwargs)
        with _replace_attribute(
            te_layernorm_linear,
            "_LayerNormLinear",
            qat_function,
        ):
            return original_forward(*args, **kwargs)

    module._verl_fp8_qat_original_forward = original_forward
    module.forward = types.MethodType(forward, module)


def _install_core_attention_kv_qat(module, state: MegatronFP8QATState):
    config = getattr(module, "config", None)
    head_dim = getattr(config, "kv_channels", None)
    if head_dim is None:
        head_dim = getattr(module, "hidden_size_per_attention_head", None)
    if head_dim is None:
        raise RuntimeError(f"Cannot infer KV head dimension for {type(module).__name__}")
    head_dim = int(head_dim)

    def hook(_module, args):
        if not state.enabled:
            return args
        if len(args) < 3:
            raise RuntimeError(
                f"{type(module).__name__} core attention expected positional query/key/value"
            )
        query, key, value = args[:3]
        return (
            query,
            STEFP8QuantKV.apply(key, head_dim),
            STEFP8QuantKV.apply(value, head_dim),
            *args[3:],
        )

    return module.register_forward_pre_hook(hook)


def apply_megatron_fp8_qat(modules, config):
    """Install 128x128-weight / 1x128-activation FP8 fake-quant on TE linears."""
    mode = str(_field(config, "mode", "")).lower()
    if mode != "w8a8_fp8":
        raise ValueError(f"Megatron fake-FP8 QAT supports only mode='w8a8_fp8', got {mode!r}")
    if str(_field(config, "scale_format", "fp32")).lower() != "fp32":
        raise ValueError("Megatron fake-FP8 QAT currently requires scale_format='fp32'")
    if str(_field(config, "scale_round", "ceil")).lower() != "ceil":
        raise ValueError("Megatron fake-FP8 QAT does not support stochastic scale rounding")
    if str(_field(config, "backward_operand_policy", "save")).lower() != "save":
        raise ValueError("Megatron fake-FP8 QAT requires backward_operand_policy='save'")

    import transformer_engine.pytorch as te

    roots = modules if isinstance(modules, list) else [modules]
    state = MegatronFP8QATState()
    ignore_patterns = list(_field(config, "ignore_patterns", []))
    converted = 0
    hook_handles = []
    kv_layers = 0

    for root in roots:
        for name, module in root.named_modules():
            if _matches_ignore(name, ignore_patterns):
                continue
            if hasattr(module, "_verl_fp8_qat_original_forward"):
                raise RuntimeError(f"Megatron FP8 QAT is already installed on {name or type(module).__name__}")
            if isinstance(module, te.LayerNormLinear):
                module.fuse_wgrad_accumulation = False
                _install_layernorm_linear(module, state)
            elif isinstance(module, te.GroupedLinear):
                module.fuse_wgrad_accumulation = False
                _install_grouped_linear(module, state)
            elif isinstance(module, te.Linear):
                module.fuse_wgrad_accumulation = False
                _install_linear(module, state)
            else:
                continue
            module._verl_fp8_qat_state = state
            converted += 1
        if bool(_field(config, "quantize_kv_cache", False)):
            for name, module in root.named_modules():
                if name.endswith("core_attention"):
                    hook_handles.append(_install_core_attention_kv_qat(module, state))
                    module._verl_fp8_qat_state = state
                    kv_layers += 1
        root._verl_fp8_qat_state = state
        root._verl_fp8_qat_hook_handles = hook_handles

    if converted == 0:
        raise RuntimeError("Megatron FP8 QAT found no Transformer Engine linear modules")
    if bool(_field(config, "quantize_kv_cache", False)) and kv_layers == 0:
        raise RuntimeError("Megatron KV fake-quant found no core_attention modules")
    return modules

