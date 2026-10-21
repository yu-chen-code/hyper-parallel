# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Static expert compute; the native dynamic expert implementation is unchanged."""

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from hyper_parallel.distributed._builder.forward_rewriter import _install_bound_forward
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, resolve_swiglu_weights
from .ep_contract import ExpertBatch


def _expert_weight_shapes(weights: tuple) -> tuple:
    """Record logical weight metadata without retaining parameter tensors."""
    return tuple(None if weight is None else tuple(weight.shape) for weight in weights)


def _validate_swiglu_shapes(shapes: tuple, local_count: int, width: int) -> None:
    """Validate logical dimensions independently of FSDP's physical storage."""
    gate, up, down = shapes
    for shape in shapes:
        if shape is None:
            continue
        if len(shape) != 3:
            raise ValueError("Graph EP expert weights must be rank-3 stacked tensors")
        if shape[0] != local_count:
            raise ValueError("Graph EP expert weights must match local_expert_count")
    if gate[-1] != width or down[1] != width:
        raise ValueError("Graph EP expert input/output width must match hidden states")
    if up is None:
        if gate[1] != 2 * down[-1]:
            raise ValueError("Graph EP fused gate/up width must be twice the down-projection input width")
    elif gate != up or gate[1] != down[-1]:
        raise ValueError("Graph EP separate gate/up weights must match the down-projection input width")


def _expert_mixed_precision(experts: Any) -> tuple:
    """Read declared hook casts without materializing or retaining parameters."""
    scheduler = getattr(experts, "hsdp_scheduler", None)
    if scheduler is not None:
        return getattr(scheduler, "mp_policy", None), getattr(scheduler, "device", None)
    get_state = getattr(experts, "_get_fsdp_state", None)
    state = get_state() if callable(get_state) else None
    group = getattr(state, "_fsdp_param_group", None)
    return getattr(group, "mp_policy", None), getattr(group, "device", None)


def _expert_compute_device(weight: torch.Tensor, declared_device: Any) -> torch.device:
    """Resolve an unindexed declared target using its resident parameter device."""
    if declared_device is None:
        return weight.device
    device = torch.device(declared_device)
    if device.type == weight.device.type and (device.index is None or device.type == "cpu"):
        return weight.device
    return device


def _expert_compute_dtype(experts: Any, hidden: torch.Tensor, weights: tuple) -> torch.dtype:
    """Validate device/linear dtype contracts, including declared hook casts."""
    policy, materialized_device = _expert_mixed_precision(experts)
    param_dtype = getattr(policy, "param_dtype", None)
    cast_inputs = getattr(policy, "cast_forward_inputs", False) and param_dtype is not None
    input_dtype = param_dtype if cast_inputs else hidden.dtype
    for weight in weights:
        if weight is not None and _expert_compute_device(weight, materialized_device) != hidden.device:
            raise ValueError("EP expert weights and hidden states must use the same compute device")
    autocast = torch.is_autocast_enabled(hidden.device.type)
    low_precision = (torch.float16, torch.bfloat16, torch.float32)
    output_dtype = input_dtype
    for weight in weights:
        if weight is None:
            continue
        dtype = param_dtype if param_dtype is not None else weight.dtype
        eligible = autocast and input_dtype in low_precision and dtype in low_precision
        if not weight.is_floating_point() or (dtype != input_dtype and not eligible):
            raise ValueError("EP expert compute dtype must match hidden states or an active mixed-precision policy")
        if eligible:
            output_dtype = torch.get_autocast_dtype(hidden.device.type)
    return getattr(policy, "output_dtype", None) or output_dtype


def validate_expert_layout(experts: Any, hidden: torch.Tensor) -> tuple:
    """Validate the logical expert and compute contracts before token exchange.

    Args:
        experts: Bound stacked SwiGLU holder, optionally managed by FSDP hooks.
        hidden: Local floating-point tokens supplied to dispatch.

    Returns:
        Current gate/up/down references for capture-time parameter naming only.

    Raises:
        ValueError: If shapes, devices or compute dtypes are incompatible.
        NotImplementedError: If the holder has no supported SwiGLU weights.
    """
    weights = resolve_swiglu_weights(experts)
    if any(not isinstance(weight, torch.Tensor) for weight in weights if weight is not None):
        raise ValueError("Graph EP expert weights must be tensors")
    managed = hasattr(experts, "hsdp_scheduler") or hasattr(experts, "_get_fsdp_state")
    shapes = getattr(experts, "ep_weight_shapes", None) if managed else None
    if shapes is None:
        shapes = _expert_weight_shapes(weights)
    _validate_swiglu_shapes(shapes, experts.local_expert_count, hidden.shape[-1])
    _expert_compute_dtype(experts, hidden, weights)
    return weights


def swiglu_expert_forward(
    experts: Any,
    states: torch.Tensor,
    index: int,
    weights: tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply one stacked SwiGLU expert in the explicit static implementation.

    Args:
        experts: Holder of the configured activation or custom fused gate.
        states: This expert's contiguous [rows, hidden] input.
        index: Local expert index in the stacked weights.
        weights: Materialized gate/up/down tensors; up is None for fused gate/up.

    Returns:
        Expert output with the same row count and hidden width as states.
    """
    gate_weight, up_weight, down_weight = weights
    if up_weight is None:
        gate_up_states = F.linear(
            states, gate_weight[index]
        )
        apply_gate = getattr(experts, "_ep_apply_gate", None)
        if apply_gate is None:
            gate_states, up_states = gate_up_states.chunk(2, dim=-1)
            activation = getattr(experts, "_ep_act_fn", F.silu)
            activated_states = activation(gate_states) * up_states
        else:
            activated_states = apply_gate(gate_up_states)
    else:
        gate_states = F.linear(
            states, gate_weight[index]
        )
        up_states = F.linear(
            states, up_weight[index]
        )
        activation = getattr(experts, "_ep_act_fn", F.silu)
        activated_states = activation(gate_states) * up_states
    return F.linear(
        activated_states, down_weight[index]
    )


def compute_expert_major(
    experts: Any,
    batch: ExpertBatch,
    weights: Optional[tuple] = None,
) -> torch.Tensor:
    """Compute static expert groups without changing the native eager kernel.

    Args:
        experts: Stacked local SwiGLU weights and activation configuration.
        batch: Expert-major input and exclusive end offsets.
        weights: Optional weights resolved and validated before graph dispatch.

    Returns:
        Expert-major output with the same shape as ``batch.tokens``.

    Note:
        Offsets are Python shape metadata; this implementation never reads
        an offsets tensor on the host. Native dynamic compute is separate.
    """
    if batch.tokens.ndim != 2:
        raise ValueError("ExpertBatch requires [rows, hidden] tokens")
    if experts.local_expert_count <= 0:
        raise ValueError("ExpertBatch requires at least one local expert")
    if not isinstance(batch.offsets, tuple):
        raise ValueError("Static ExpertBatch offsets must be a tuple of integers")
    ends = batch.offsets
    if len(ends) != experts.local_expert_count or ends[-1] != batch.tokens.shape[0]:
        raise ValueError("ExpertBatch offsets do not cover the local expert rows")
    if any(not isinstance(end, int) or isinstance(end, bool) for end in ends):
        raise ValueError("ExpertBatch offsets must contain integers")
    if any(end < start for start, end in zip((0, *ends[:-1]), ends)):
        raise ValueError("ExpertBatch offsets must be monotonic")
    if weights is None:
        weights = resolve_swiglu_weights(experts)
    outputs = []
    start = 0
    for index, end in enumerate(ends):
        outputs.append(swiglu_expert_forward(experts, batch.tokens[start:end], index, weights))
        start = end
    return torch.cat(outputs, dim=0)


def _static_expert_forward(experts: Any, states: torch.Tensor, *, offsets: tuple[int, ...]) -> torch.Tensor:
    """Enter compute through the module boundary, preserving graph parameter access."""
    return compute_expert_major(experts, ExpertBatch(states, offsets))


def bind_static_expert_forward(module: Any, ep_size: int, *, apply_gate: Optional[Callable] = None) -> None:
    """Bind a dedicated graph model's local expert module.

    Args:
        module: MoE block with stacked experts; use a separate model for eager.
        ep_size: Positive expert-parallel group size dividing the expert count.
        apply_gate: Model-owned fused gate, preserving activation semantics.

    Raises:
        ValueError: If the existing binder rejects the expert contract.

    Note:
        This explicit graph-only operation replaces this holder's forward.
        Native eager factories and their default execution paths never call it.
    """
    bind_local_expert_forward(module, ep_size, apply_gate=apply_gate)
    experts = module.experts
    experts.ep_weight_shapes = _expert_weight_shapes(resolve_swiglu_weights(experts))
    _install_bound_forward(experts, _static_expert_forward)
