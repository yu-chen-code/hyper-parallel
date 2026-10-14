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
"""Fixed-slot token routing shared by eager execution and graph capture."""

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as fc
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._python_dispatch import _disable_current_modes

from .dispatch_contract import DispatchMetadata, DispatchPlan, DispatchResult, ExpertBatch, RoutingResult

_MAX_DENSE_DISPATCH_BYTES = 256 * 1024 * 1024


def _check_routing_tensors(hidden: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> None:
    """Reject incompatible routing operands without reading device values."""
    if not hidden.is_floating_point() or not weights.is_floating_point():
        raise ValueError("EP hidden states and Router weights must be floating-point tensors")
    if indices.device != hidden.device or weights.device != hidden.device:
        raise ValueError("EP hidden states, Router indices and Router weights must use the same device")


def ep_all_to_all(
    tokens: torch.Tensor,
    send_counts: Sequence[int],
    recv_counts: Sequence[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Exchange EP token chunks with statically known split sizes.

    Splits are source-rank ordered Python integers, fixed for every replay.
    The caller must derive a matching plan on all ranks.

    Args:
        tokens: Local token tensor ordered by destination rank.
        send_counts: Number of local rows sent to each rank.
        recv_counts: Number of rows received from each rank.
        group: EP process group.

    Returns:
        Received tokens ordered by source rank, with autograd support.

    Raises:
        RuntimeError: If distributed communication is not initialized.
        ValueError: If the token shape or static split sizes are invalid.

    Note:
        Changing split sizes after graph capture requires recompilation.
    """
    if tokens.ndim < 1:
        raise ValueError("EP tokens must have a leading token dimension")
    if not dist.is_initialized():
        raise RuntimeError("EP all-to-all requires an initialized process group")

    group_size = dist.get_world_size(group)
    for name, counts in (("send_counts", send_counts), ("recv_counts", recv_counts)):
        if len(counts) != group_size or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in counts
        ):
            raise ValueError(
                f"{name} must contain {group_size} non-negative Python integers"
            )
    if tokens.shape[0] != sum(send_counts):
        raise ValueError(
            f"EP token rows ({tokens.shape[0]}) must equal "
            f"sum(send_counts) ({sum(send_counts)})"
        )

    return fc.all_to_all_single_autograd(
        tokens.contiguous(), list(recv_counts), list(send_counts), group
    )


def _expert_layout(experts: Any, group_size: int) -> tuple[int, int]:
    """Resolve the global count from the eager binder's local shard size."""
    local_count = getattr(experts, "local_expert_count", None)
    if not isinstance(local_count, int) or local_count <= 0:
        raise ValueError("Graph EP requires a positive local_expert_count from the eager binder")
    expert_count = local_count * group_size
    declared_count = getattr(experts, "num_experts", expert_count)
    if declared_count != expert_count:
        raise ValueError("Graph EP expert count conflicts with the eager binder's local_expert_count")
    return local_count, expert_count


def _check_routing(
    experts: Any,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[int, int, int, int, int]:
    """Check the static EP layout and return its fixed dimensions."""
    if not dist.is_initialized():
        raise RuntimeError("Static EP routing requires an initialized process group")
    _check_routing_tensors(hidden_states, topk_indices, topk_weights)
    if hidden_states.ndim < 2 or topk_indices.shape != topk_weights.shape:
        raise ValueError("Static EP requires hidden states and matching Top-K tensors")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Top-K expert indices must be integer tensors")
    if hidden_states.shape[-1] < 1:
        raise ValueError("Static EP requires nonempty Top-K and hidden dimensions")
    token_count = hidden_states.numel() // hidden_states.shape[-1]
    if topk_indices.ndim != 2 or topk_indices.shape[0] != token_count:
        raise ValueError("Top-K routing must have one row per hidden-state token")
    group_size = dist.get_world_size(group)
    local_count, expert_count = _expert_layout(experts, group_size)
    if topk_indices.shape[-1] < 1 or hidden_states.shape[-1] < 1:
        raise ValueError("Static EP requires nonempty Top-K and hidden dimensions")
    dispatch_bytes = expert_count * topk_indices.numel() * hidden_states.shape[-1] * hidden_states.element_size()
    if dispatch_bytes > _MAX_DENSE_DISPATCH_BYTES:
        raise ValueError(
            f"Static EP dense dispatch requires {dispatch_bytes} bytes per tensor, "
            f"exceeding the {_MAX_DENSE_DISPATCH_BYTES}-byte validation limit"
        )
    # Concrete calls can reject bad inputs before communication. During graph
    # capture the values are FakeTensors; the replay path marks an invalid
    # route nonfinite with device ops rather than synchronizing to the host.
    if getattr(experts, "validation_mode", "strict") == "strict" and not isinstance(topk_indices, FakeTensor):
        valid = ((topk_indices >= 0) & (topk_indices < expert_count)).all()
        if not bool(valid.item()):
            raise ValueError("Top-K expert IDs must be in [0, num_experts)")
    return group_size, local_count, expert_count, topk_indices.numel(), hidden_states.shape[-1]


def _agree_routing(
    signature: Optional[tuple],
    error: Optional[str],
    group: dist.ProcessGroup,
) -> None:
    """Share validation failures and communication layouts before all-to-all."""
    plans = [None] * dist.get_world_size(group)
    # Strict validation is a diagnostic synchronization; capture keeps it
    # outside the FX graph. Local eager validation never calls this gate.
    with _disable_current_modes():
        dist.all_gather_object(plans, (signature, error), group=group)
    failures = [f"rank {rank}: {plan[1]}" for rank, plan in enumerate(plans) if plan[1]]
    if failures:
        raise ValueError("Graph EP routing validation failed: " + "; ".join(failures))
    signatures = [plan[0] for plan in plans]
    capacities = [peer[0] for peer in signatures]
    if any(capacity != signature[0] for capacity in capacities):
        raise ValueError(
            "Graph EP requires the same local token count times Top-K on every "
            f"EP rank during capture; received capacities {capacities}"
        )
    if any(peer != signature for peer in signatures):
        raise ValueError(
            "Graph EP requires identical routing signatures (slots, global experts, "
            f"local experts, hidden width, dtype, compute metadata); received {signatures}"
        )


def routing_preflight(group: Any, prepare: Callable[[], DispatchPlan]) -> DispatchPlan:
    """Agree on local preparation errors before any token exchange.

    Args:
        group: EP participants entering the same metadata collective.
        prepare: Local Router and contract checks producing a per-call plan.

    Returns:
        This rank's plan after every peer has accepted the communication layout.

    Raises:
        ValueError: If any prepare fails or the peer signatures disagree.
        RuntimeError: If distributed communication is not initialized.
    """
    if not dist.is_initialized():
        raise RuntimeError("Static EP routing requires an initialized process group")
    plan, error = None, None
    try:
        plan = prepare()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    _agree_routing(plan.signature if plan is not None else None, error, group)
    if plan is None:
        raise ValueError(error)
    return plan


def reject_routing(group: Any, message: str) -> None:
    """Reject an unsupported entry through the group preparation gate.

    Args:
        group: EP peers entering this call or a normal routed preparation.
        message: Local configuration failure to share with all peers.

    Raises:
        ValueError: Always, after the group has exchanged failure status.
    """
    def prepare() -> DispatchPlan:
        """Represent the unsupported entry as a local preparation failure."""
        raise ValueError(message)

    routing_preflight(group, prepare)


@dataclass(frozen=True)
class FixedSlotTokenDispatcher:
    """Per-expert padding shared by eager execution and graph capture.

    Args:
        group: EP participants, identical for dispatch and combine.
        local_expert_count: Number of complete local experts before EDP sharding.
        validation_mode: ``strict`` agrees on every eager preparation and reads
            concrete IDs; ``local`` performs metadata-only local checks without
            host synchronization. Capture always agrees across ranks. Local
            mode requires a previously validated, consistent rank configuration.
    """

    group: Any
    local_expert_count: int
    validation_mode: str = "strict"
    supports_graph_capture: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Reject an unknown validation policy at configuration time."""
        if self.validation_mode not in ("strict", "local"):
            raise ValueError("EP validation_mode must be 'strict' or 'local'")

    def prepare(self, hidden_states: torch.Tensor, routing: RoutingResult) -> DispatchPlan:
        """Prepare a local static plan without token communication.

        Args:
            hidden_states: Floating-point local tokens, with a trailing hidden axis.
            routing: Matching Top-K IDs and floating-point Router weights.

        Returns:
            A per-call plan owning routing tensors and static combine metadata.

        Raises:
            ValueError: On incompatible devices, dtypes, shapes, IDs or budget.
            RuntimeError: If distributed communication is not initialized.
        """
        size, local, experts, slots, width = _check_routing(
            self, hidden_states, routing.indices, routing.weights, self.group
        )
        metadata = DispatchMetadata(tuple(hidden_states.shape), size, local, slots,
                                    (local * slots,) * size, routing)
        return DispatchPlan(metadata, (slots, experts, local, width, str(hidden_states.dtype)))

    def dispatch(self, hidden_states: torch.Tensor, plan: DispatchPlan) -> DispatchResult:
        """Exchange padded slots and return local expert-major rows.

        Args:
            hidden_states: Tokens used to prepare this invocation's plan.
            plan: Local plan agreed by all EP peers in strict/capture mode.

        Returns:
            ExpertBatch and matching metadata for this call's combine.
        """
        metadata = plan.metadata
        size, local, slots = metadata.group_size, metadata.local_expert_count, metadata.slots
        width = metadata.input_shape[-1]
        indices = metadata.routing.indices
        topk = indices.shape[-1]
        flat_tokens = hidden_states.reshape(-1, width)
        repeated = flat_tokens.unsqueeze(1).expand(-1, topk, -1).reshape(slots, width)
        expert_ids = torch.arange(size * local, device=indices.device)
        selected = indices.reshape(1, -1) == expert_ids.reshape(-1, 1)
        dispatched = (selected.unsqueeze(-1) * repeated).reshape(size * local * slots, width)
        counts = metadata.split_sizes
        received = ep_all_to_all(dispatched, counts, counts, self.group)
        received = received.reshape(size, local, slots, width)
        expert_size = size * slots
        tokens = received.permute(1, 0, 2, 3).contiguous().reshape(local * expert_size, width)
        offsets = tuple(expert_size * (index + 1) for index in range(local))
        return DispatchResult(ExpertBatch(tokens, offsets), metadata)

    def combine(self, expert_output: torch.Tensor, metadata: DispatchMetadata) -> torch.Tensor:
        """Return expert outputs and aggregate Router-weighted slots.

        Args:
            expert_output: Expert-major output matching the dispatched batch.
            metadata: Unmodified metadata returned by the corresponding dispatch.

        Returns:
            Local tokens restored to the input shape; invalid IDs mark NaNs.
        """
        size, local, slots = metadata.group_size, metadata.local_expert_count, metadata.slots
        width = metadata.input_shape[-1]
        rank_major = expert_output.reshape(local, size, slots, width).permute(1, 0, 2, 3).contiguous()
        counts = metadata.split_sizes
        combined = ep_all_to_all(rank_major.reshape(-1, width), counts, counts, self.group)
        routed = combined.reshape(size, local, slots, width).sum(dim=(0, 1))
        indices, weights = metadata.routing.indices, metadata.routing.weights
        weights = weights.to(dtype=routed.dtype)
        weighted = (routed * weights.reshape(slots, 1)).reshape(*metadata.input_shape[:-1],
                                                             indices.shape[-1], width)
        output = weighted.sum(dim=-2)
        valid = ((indices >= 0) & (indices < size * local)).all()
        return torch.where(valid, output, torch.full_like(output, float("nan")))
