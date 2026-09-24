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

"""QAT (Quantization-Aware Training) utilities for verl FSDP training."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch.nn as nn

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class QATConfig(BaseConfig):
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None
    # FP8 scale format, must match the rollout: "fp32", "e8m0" (DeepGEMM), or "auto" (VLLM_USE_DEEP_GEMM_E8M0)
    scale_format: str = "auto"

    # E8M0 scale rounding: "ceil", "stochastic_weight", or "stochastic"
    scale_round: str = "ceil"

    # Fake-quantize K/V per token and head; enable only with rollout kv_cache_dtype=fp8_per_token_head
    quantize_kv_cache: bool = False

    # FP8 QAT backward operands: "save" keeps QDQ tensors, "recompute" rematerializes them
    backward_operand_policy: str = "save"


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        raise ValueError("quantization_config_path is required when QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    if qat_config.ignore_patterns:
        original_ignore = quant_config.get("ignore", [])
        quant_config["ignore"] = qat_config.ignore_patterns
        if original_ignore != qat_config.ignore_patterns:
            logger.info(f"Overriding JSON 'ignore' field: {original_ignore} -> {qat_config.ignore_patterns}")

    logger.info("Successfully loaded QAT quantization config")
    return quant_config


def _should_quantize(name: str, module: nn.Module, config: QATConfig) -> bool:
    """Check if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.match(regex, name):
                logger.debug(f"Ignoring {name} due to regex pattern: {regex}")
                return False
        else:
            if pattern in name:
                logger.debug(f"Ignoring {name} due to pattern: {pattern}")
                return False

    # fp8 modes (w8a8_fp8 / w8a16_fp8) use 128-wide blocks, not the NVFP4 group_size.
    divisor = 128 if str(config.mode).lower().endswith("_fp8") else config.group_size
    if module.in_features % divisor != 0:
        logger.warning(f"Skipping {name}: in_features={module.in_features} not divisible by {divisor}")
        return False

    return True


def apply_qat(
    model: nn.Module,
    config: QATConfig | dict[str, Any],
) -> nn.Module:
    """Apply QAT to a model by replacing nn.Linear with QATLinear."""
    from verl.utils.qat.linear import QATLinear, QATMode

    if not isinstance(config, QATConfig):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    mode = QATMode(config.mode.lower())
    logger.info(
        f"Applying QAT with mode={mode.value}, group_size={config.group_size}, "
        f"scale_format={getattr(config, 'scale_format', 'auto')}, "
        f"scale_round={getattr(config, 'scale_round', 'ceil')}, "
        f"backward_operand_policy={getattr(config, 'backward_operand_policy', 'save')}"
    )

    modules_to_replace = []
    for name, module in model.named_modules():
        if _should_quantize(name, module, config):
            modules_to_replace.append((name, module))

    logger.info(f"Found {len(modules_to_replace)} Linear layers to convert to QAT")

    converted_count = 0
    for name, module in modules_to_replace:
        if isinstance(module, QATLinear):
            continue

        fake_quant_module = QATLinear.from_linear(
            module,
            mode=mode,
            group_size=config.group_size,
            activation_observer=config.activation_observer,
            scale_format=getattr(config, "scale_format", "auto"),
            scale_round=getattr(config, "scale_round", "ceil"),
            backward_operand_policy=getattr(
                config,
                "backward_operand_policy",
                "save",
            ),
        )

        _set_module(model, name, fake_quant_module)
        converted_count += 1

    logger.info(f"Successfully applied QAT to {converted_count} layers")

    if getattr(config, "quantize_kv_cache", False):
        apply_kv_cache_fake_quant(model, config)

    return model


def apply_kv_cache_fake_quant(model: nn.Module, config: "QATConfig") -> int:
    """Fake-quantize K and V at the points where the fp8_per_token_head rollout quantizes them.

    V is quantized at the projection output and K after k_norm and RoPE.
    """
    import sys

    head_dim = _infer_head_dim(model)
    if head_dim is None:
        raise RuntimeError("KV fake-quant is enabled but head_dim could not be inferred")

    if getattr(model, "_kv_fake_quant_applied", False):
        raise RuntimeError("KV fake-quant is already installed on this model")

    switch = _KVFakeQuantSwitch()

    hooked_v = 0
    for name, module in model.named_modules():
        if name.endswith("v_proj"):
            module.register_forward_hook(_v_quant_hook(switch, head_dim))
            hooked_v += 1

    hooked_k = 0
    for name, module in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        modeling = sys.modules.get(type(module).__module__)
        if modeling is None or not hasattr(modeling, "apply_rotary_pos_emb"):
            raise RuntimeError(
                f"KV fake-quant is enabled but {type(module).__module__} exposes no apply_rotary_pos_emb"
            )
        module.forward = _rope_quant_wrapper(module, modeling, head_dim, switch)
        hooked_k += 1

    if hooked_k != hooked_v:
        raise RuntimeError(f"KV fake-quant hooked {hooked_k} K projections but {hooked_v} V projections")

    model._kv_fake_quant_applied = True
    model._kv_fake_quant_switch = switch
    logger.info(
        f"KV cache fake-quant (per-token-head e4m3) on {hooked_k} layers, head_dim={head_dim} "
        "(K after RoPE, V on v_proj output)"
    )
    return hooked_k + hooked_v


class _KVFakeQuantSwitch:
    """One on/off flag shared by every KV hook installed on a model."""

    __slots__ = ("enabled",)

    def __init__(self):
        self.enabled = True


def _v_quant_hook(switch: "_KVFakeQuantSwitch", head_dim: int):
    """Fake-quantize the v_proj output."""
    from verl.utils.qat.linear import STEFP8QuantKV

    def hook(_module, _inputs, output):
        if not switch.enabled:
            return output
        return STEFP8QuantKV.apply(output, head_dim)

    return hook


def _rope_quant_wrapper(attn_module: nn.Module, modeling, head_dim: int, switch: "_KVFakeQuantSwitch"):
    """Run this attention module with a RoPE that fake-quantizes K; other models keep the original."""
    from verl.utils.qat.linear import STEFP8QuantKV

    original_forward = attn_module.forward

    def forward(*args, **kwargs):
        if not switch.enabled:
            return original_forward(*args, **kwargs)

        original_rope = modeling.apply_rotary_pos_emb

        def rope_then_quant(q, k, cos, sin, **rope_kwargs):
            q_embed, k_embed = original_rope(q, k, cos, sin, **rope_kwargs)
            return q_embed, STEFP8QuantKV.apply(k_embed, head_dim)

        modeling.apply_rotary_pos_emb = rope_then_quant
        try:
            return original_forward(*args, **kwargs)
        finally:
            modeling.apply_rotary_pos_emb = original_rope

    return forward


def _infer_head_dim(model: nn.Module) -> Optional[int]:
    """Read head_dim from the k_proj weight."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", None)
    from_weight = None
    for name, module in model.named_modules():
        if name.endswith("k_proj") and hasattr(module, "out_features") and kv_heads:
            from_weight = int(module.out_features) // int(kv_heads)
            break

    declared = getattr(cfg, "head_dim", None)
    declared = int(declared) if declared else None

    if from_weight and declared and from_weight != declared:
        raise RuntimeError(f"config.head_dim={declared} disagrees with k_proj weight ({from_weight})")
    return from_weight or declared


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


FUSION_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def setup_fusion_siblings(model: nn.Module):
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    from verl.utils.qat.linear import QATLinear

    qat_modules = {name: m for name, m in model.named_modules() if isinstance(m, QATLinear)}

    counts = {}
    for group_name, suffixes in FUSION_PATTERNS.items():
        groups: dict[str, dict[str, nn.Module]] = {}
        for name, module in qat_modules.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    parent = name.rsplit(".", 1)[0]
                    groups.setdefault(parent, {})[suffix] = module

        count = 0
        for parent, projs in groups.items():
            if len(projs) >= 2:
                modules = list(projs.values())
                for i, m in enumerate(modules):
                    siblings = modules[:i] + modules[i + 1 :]
                    m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
                count += 1
        counts[group_name] = count

    logger.info(f"[QAT Fuse] Setup fusion siblings: {counts}")
    return counts


def enable_qat_fuse(model: nn.Module):
    """Enable QAT fuse mode: sets up fusion siblings for weight scale fusion."""
    setup_fusion_siblings(model)
    model._qat_fuse_enabled = True
    logger.info("[QAT Fuse] Enabled QAT fuse mode")


def invalidate_all_scales(model: nn.Module):
    """Clear all cached weight scales after optimizer.step()."""
    from verl.utils.qat.linear import QATLinear

    count = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module._weight_blockwise_scale = None
            module._weight_global_scale = None
            module._cached_weight_amax = None
            count += 1

    logger.debug(f"[QAT Fuse] Invalidated scales for {count} QATLinear layers")
