# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from collections.abc import Callable, Mapping

import torch
import torch.nn as nn

try:
    from megatron.bridge import AutoBridge
except ImportError:
    # `pip install verl[mcore]` or
    print("Megatron-Bridge package not found. Please install Megatron-Bridge with `pip install megatron-bridge`")
    raise

try:
    from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model
except ImportError:
    from megatron.core import parallel_state, tensor_parallel

    class LinearForLastLayer(nn.Linear):
        """Replicated projection head used by value models."""

        def __init__(
            self,
            input_size: int,
            output_size: int,
            sequence_parallel: bool,
            bias: bool = False,
            dropout: float = 0.0,
            *,
            output_in_fp32: bool = True,
            tp_group: torch.distributed.ProcessGroup | None = None,
            init_method: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
            perform_initialization: bool = True,
        ) -> None:
            self._init_method = init_method
            self.perform_initialization = perform_initialization
            super().__init__(in_features=input_size, out_features=output_size, bias=bias)
            self.sequence_parallel = sequence_parallel
            self.dropout = nn.Dropout(dropout)
            self.output_in_fp32 = output_in_fp32
            self.tp_group = tp_group
            if sequence_parallel:
                setattr(self.weight, "sequence_parallel", True)
            if self.bias is not None:
                setattr(self.bias, "sequence_parallel", True)

        def reset_parameters(self) -> None:
            if not getattr(self, "perform_initialization", True):
                return
            init_method = getattr(self, "_init_method", None)
            if init_method is None:
                super().reset_parameters()
            else:
                init_method(self.weight)
                if self.bias is not None:
                    nn.init.zeros_(self.bias)

        def forward(
            self,
            input_: torch.Tensor,
            weight: torch.Tensor | None = None,
            runtime_gather_output: bool | None = None,
        ) -> tuple[torch.Tensor, None]:
            del weight, runtime_gather_output
            output = super().forward(self.dropout(input_))
            if self.output_in_fp32:
                output = output.float()
            if self.sequence_parallel:
                output = tensor_parallel.gather_from_sequence_parallel_region(
                    output,
                    tensor_parallel_output_grad=False,
                    group=self.tp_group,
                )
            return output, None

    def _ensure_model_list(model):
        return model if isinstance(model, list) else [model]

    def make_value_model(hidden_size: int, sequence_parallel: bool):
        """Create a Bridge pre-wrap hook that installs a scalar value head."""

        def hook(model):
            model_chunks = _ensure_model_list(model)
            virtual_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            if parallel_state.get_pipeline_model_parallel_world_size() > 1 and virtual_size is not None:
                post_process = [
                    parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
                    for vp_stage in range(virtual_size)
                ]
            else:
                post_process = [parallel_state.is_pipeline_last_stage()]
            if len(post_process) != len(model_chunks):
                raise ValueError(
                    "Model list length and pipeline post-process list length must match: "
                    f"{len(model_chunks)} chunks versus {len(post_process)} flags"
                )
            for index, model_chunk in enumerate(model_chunks):
                if post_process[index]:
                    model_chunk.output_layer = LinearForLastLayer(
                        input_size=hidden_size,
                        output_size=1,
                        sequence_parallel=sequence_parallel,
                        tp_group=getattr(getattr(model_chunk, "pg_collection", None), "tp", None),
                    )
            return model_chunks

        return hook

    def freeze_moe_router(model):
        """Freeze routed and shared-expert gate parameters."""
        model_chunks = _ensure_model_list(model)
        for model_chunk in model_chunks:
            decoder = getattr(model_chunk, "decoder", None)
            for layer in getattr(decoder, "layers", ()):
                mlp = getattr(layer, "mlp", None)
                if mlp is None:
                    continue
                router = getattr(mlp, "router", None)
                for name in ("weight", "bias"):
                    parameter = getattr(router, name, None)
                    if parameter is not None:
                        parameter.requires_grad = False
                shared_experts = getattr(mlp, "shared_experts", None)
                for name in ("gate_weight", "gate_bias"):
                    parameter = getattr(shared_experts, name, None)
                    if parameter is not None:
                        parameter.requires_grad = False
        return model_chunks

try:
    from megatron.bridge.training.utils.config_utils import create_ddp_config
except ImportError:

    def create_ddp_config(
        wrap_with_ddp: bool = True,
        use_distributed_optimizer: bool = True,
        use_megatron_fsdp: bool = False,
        overrides: Mapping[str, object] | None = None,
        finalize: bool = True,
    ):
        """Build the DDP config expected by Bridge external model construction."""
        if not wrap_with_ddp:
            return None

        from megatron.bridge.training.config import DistributedDataParallelConfig

        values = {"use_distributed_optimizer": use_distributed_optimizer}
        if use_megatron_fsdp:
            values.update(
                {
                    "use_distributed_optimizer": True,
                    "check_for_nan_in_grad": True,
                    "use_megatron_fsdp": True,
                    "data_parallel_sharding_strategy": "optim_grads_params",
                    "overlap_grad_reduce": True,
                }
            )
        values.update(overrides or {})
        config = DistributedDataParallelConfig(**values)
        if finalize:
            config.finalize()
        return config


def apply_provider_overrides_and_finalize(
    provider,
    *,
    dtype: torch.dtype | None = None,
    overrides: Mapping[str, object] | None = None,
):
    """Backport the external-trainer provider helper added after Bridge 0.3.1."""
    native = getattr(provider, "apply_overrides_and_finalize", None)
    if native is not None:
        return native(dtype=dtype, overrides=overrides)

    if dtype is not None:
        provider.params_dtype = dtype
        provider.fp16 = dtype == torch.float16
        provider.bf16 = dtype == torch.bfloat16
    for name, value in (overrides or {}).items():
        if not hasattr(provider, name):
            raise AttributeError(f"{type(provider).__name__} has no attribute {name!r}")
        setattr(provider, name, value)
    provider.finalize()
    return provider

__all__ = [
    "AutoBridge",
    "LinearForLastLayer",
    "freeze_moe_router",
    "make_value_model",
    "create_ddp_config",
    "apply_provider_overrides_and_finalize",
]
