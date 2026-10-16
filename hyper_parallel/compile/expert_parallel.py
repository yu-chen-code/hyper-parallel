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
"""Graph-capturable EP token exchange and padded static-shape routing.

The caller retains the eager EP layout contract: sends are destination-rank
ordered and receives are source-rank ordered, with one Python split size per
rank. The routed path preserves eager SwiGLU expert weights and gate semantics,
while padding each expert to a fixed slot count for graph replay.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as fc
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._python_dispatch import _disable_current_modes

from hyper_parallel.distributed.expert_parallel.dispatch_contract import ExpertBatch
from hyper_parallel.distributed.expert_parallel.experts import (
    _reset_graph_routed_forward,
    _set_graph_routed_forward,
    compute_expert_major,
)

__all__ = ["ep_all_to_all", "routed_forward"]

_MAX_DENSE_DISPATCH_BYTES = 256 * 1024 * 1024
_CAPTURE_EP_GROUPS: ContextVar[Optional[list]] = ContextVar("capture_ep_groups", default=None)


@contextmanager
def graph_ep_trace(enabled: bool) -> Iterator[list]:
    """Use static routing during capture and collect its actual EP groups."""
    groups = []
    if not enabled:
        yield groups
        return
    token = _set_graph_routed_forward(routed_forward, _agree_routing_error)
    group_token = _CAPTURE_EP_GROUPS.set(groups)
    try:
        yield groups
    finally:
        _CAPTURE_EP_GROUPS.reset(group_token)
        _reset_graph_routed_forward(token)


def ep_all_to_all(
    tokens: torch.Tensor,
    send_counts: Sequence[int],
    recv_counts: Sequence[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Exchange EP token chunks with statically known split sizes.

    ``send_counts[peer]`` and ``recv_counts[peer]`` describe the same
    source-rank ordered layout as eager EP's ``ep_all_to_all``. The counts must
    be Python integers fixed for every replay of the captured graph. The
    caller is responsible for deriving a matching plan on all ranks.

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

    groups = _CAPTURE_EP_GROUPS.get()
    if groups is not None and all(peer is not group for peer in groups):
        groups.append(group)
    return fc.all_to_all_single_autograd(
        tokens.contiguous(), list(recv_counts), list(send_counts), group
    )


def _expert_layout(experts: torch.nn.Module, group_size: int) -> tuple[int, int]:
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
    experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[int, int, int, int, int]:
    """Check the static EP layout and return its fixed dimensions."""
    if not dist.is_initialized():
        raise RuntimeError("Static EP routing requires an initialized process group")
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
    if not isinstance(topk_indices, FakeTensor):
        valid = ((topk_indices >= 0) & (topk_indices < expert_count)).all()
        if not bool(valid.item()):
            raise ValueError("Top-K expert IDs must be in [0, num_experts)")
    return group_size, local_count, expert_count, topk_indices.numel(), hidden_states.shape[-1]


def _agree_routing(
    signature: Optional[tuple[int, int, int, int, str]],
    error: Optional[str],
    group: dist.ProcessGroup,
) -> None:
    """Share validation failures and communication layouts before all-to-all."""
    plans = [None] * dist.get_world_size(group)
    # This one-time metadata exchange belongs to capture, not the FX graph.
    # Exclude its tensor serialization from FakeTensor/proxy tracing.
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
            f"local experts, hidden width, dtype); received {signatures}"
        )


def _agree_routing_error(group: dist.ProcessGroup, error: str) -> None:
    """Send an unsupported routing path through the normal EP preflight."""
    _agree_routing(None, error, group)


def routed_forward(
    experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run routed SwiGLU experts with fixed-size EP communication.

    Each expert receives one slot for every local token and Top-K position.
    Unselected slots contain zero, so route changes do not alter collective
    shapes. ``experts`` must use the eager EP binder's local weight contract.

    Args:
        experts: Local stacked SwiGLU experts initialized by the eager EP binder.
        hidden_states: Local token states with a trailing hidden dimension.
        topk_indices: Integer expert IDs in ``[0, local_expert_count * ep_size)`` shaped
            ``[local_tokens, top_k]``.
        topk_weights: Router weights with the same shape as ``topk_indices``.
        group: Expert-parallel process group.

    Returns:
        Routed expert output with the same shape as ``hidden_states``.

    Raises:
        RuntimeError: If distributed communication is not initialized.
        ValueError: If the static token layout or a concrete expert ID is invalid.

    Note:
        Expert activation must map zero to zero. This first-stage path trades
        memory and compute for static shapes; changing the token count requires
        graph recompilation. A single dense dispatch tensor is limited to
        256 MiB per rank for this validation-scale implementation; the full
        training step requires additional memory.
        An invalid expert ID on graph replay makes the output NaN using device
        operations; no per-step host synchronization is used for the check.
    """
    if not dist.is_initialized():
        raise RuntimeError("Static EP routing requires an initialized process group")
    try:
        layout = _check_routing(experts, hidden_states, topk_indices, topk_weights, group)
        local_error = None
    except (ValueError, RuntimeError) as error:
        layout = None
        local_error = str(error)
    signature = None
    if layout is not None:
        signature = (layout[3], layout[2], layout[1], layout[4], str(hidden_states.dtype))
    _agree_routing(signature, local_error, group)
    if layout is None:
        raise ValueError(local_error)
    group_size, local_count, expert_count, slots, width = layout
    flat_tokens = hidden_states.reshape(-1, width)
    topk = topk_indices.shape[-1]
    repeated = flat_tokens.unsqueeze(1).expand(-1, topk, -1).reshape(slots, width)
    expert_ids = torch.arange(expert_count, device=topk_indices.device)
    selected = topk_indices.reshape(1, -1) == expert_ids.reshape(-1, 1)
    dispatched = (selected.unsqueeze(-1) * repeated).reshape(
        group_size * local_count * slots, width
    )
    counts = [local_count * slots] * group_size
    received = ep_all_to_all(dispatched, counts, counts, group)
    received = received.reshape(group_size, local_count, slots, width)
    expert_size = group_size * slots
    expert_rows = received.permute(1, 0, 2, 3).contiguous().reshape(
        local_count * expert_size, width
    )
    offsets = tuple(expert_size * (index + 1) for index in range(local_count))
    batch = ExpertBatch(expert_rows, offsets)
    expert_output = compute_expert_major(experts, batch)
    rank_major_output = expert_output.reshape(
        local_count, group_size, slots, width
    ).permute(1, 0, 2, 3).contiguous()
    combined = ep_all_to_all(
        rank_major_output.reshape(-1, width), counts, counts, group
    )
    routed = combined.reshape(group_size, local_count, slots, width).sum(dim=(0, 1))
    # Eager EP casts router weights to the hidden dtype before aggregation.
    router_weights = topk_weights.to(dtype=routed.dtype)
    weighted = (routed * router_weights.reshape(slots, 1)).reshape(
        *hidden_states.shape[:-1], topk, width
    )
    output = weighted.sum(dim=-2)
    valid = ((topk_indices >= 0) & (topk_indices < expert_count)).all()
    return torch.where(valid, output, torch.full_like(output, float("nan")))
