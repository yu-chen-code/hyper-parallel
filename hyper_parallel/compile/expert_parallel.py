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
"""Capture records and compatibility wrappers for shared expert routing."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import torch
import torch.distributed as dist

from hyper_parallel.distributed.expert_parallel.dispatch_contract import DispatchPlan, RoutingResult
from hyper_parallel.distributed.expert_parallel.dispatchers import ep_all_to_all, reject_routing, routing_preflight
from hyper_parallel.distributed.expert_parallel.experts import (
    _reset_graph_ep_capture,
    _set_graph_ep_capture,
    fixed_slot_routed_forward,
)

__all__ = ["ep_all_to_all", "routed_forward"]


@dataclass
class GraphEPCaptureContext:
    """Own peer agreement, group/parameter records, and scoped capture lifetime."""

    model: Optional[torch.nn.Module] = None
    groups: list = field(default_factory=list)
    expert_param_names: set = field(default_factory=set)

    def preflight(self, group: dist.ProcessGroup, prepare: Callable[[], DispatchPlan]) -> DispatchPlan:
        """Agree on preparation and record the accepted EP participants.

        Args:
            group: EP participants entering this routed call.
            prepare: Local preparation callback, including Router and contracts.

        Returns:
            Local plan accepted by all peers.

        Raises:
            ValueError: On any peer's preparation error or signature mismatch.
        """
        plan = routing_preflight(group, prepare)
        if all(peer is not group for peer in self.groups):
            self.groups.append(group)
        return plan

    def record_weights(self, weights: tuple) -> None:
        """Identify actual expert parameters without retaining their tensors.

        Args:
            weights: Current materialized/functional gate, optional up, and down
                references, used only to resolve exact model parameter names.
        """
        if self.model is None:
            return
        weight_ids = {id(weight) for weight in weights if weight is not None}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            if id(parameter) in weight_ids:
                self.expert_param_names.add(name)

    def reject_backend(self, group: dist.ProcessGroup, message: str) -> None:
        """Share an unsupported backend/path error before any data exchange."""
        reject_routing(group, message)

    def reject_split_dispatch(self, group: dist.ProcessGroup) -> None:
        """Reject the eager overlap schedule through the same preparation gate."""
        self.reject_backend(group, "Graph EP does not support split dispatch with overlap_shared_expert=True")


@contextmanager
def graph_ep_trace(enabled: bool, model: Optional[torch.nn.Module] = None) -> Iterator[GraphEPCaptureContext]:
    """Scope capture records and restore the parent even when preparation fails.

    Args:
        enabled: Install the context for shared EP entry points when true.
        model: Functional model used to resolve exact expert parameter names.

    Yields:
        Capture record consumed by GraphCompiler; never selects a dispatcher.
    """
    context = GraphEPCaptureContext(model)
    if not enabled:
        yield context
        return
    token = _set_graph_ep_capture(context)
    try:
        yield context
    finally:
        _reset_graph_ep_capture(token)


def routed_forward(experts: torch.nn.Module, hidden_states: torch.Tensor,
                   topk_indices: torch.Tensor, topk_weights: torch.Tensor,
                   group: dist.ProcessGroup) -> torch.Tensor:
    """Delegate the explicit fixed-slot API to the common module-based pipeline.

    The holder uses the eager binder's local expert count and stacked weights.
    This explicit API selects fixed_slot itself; GraphCompiler never changes
    a model's selected dispatcher. Invalid replay IDs produce a nonfinite
    output, and GraphCompiler synchronizes finite loss before optimizer use.

    Args:
        experts: Stacked local SwiGLU holder compatible with the eager binder.
        hidden_states: Local floating-point tokens with a trailing hidden axis.
        topk_indices: Integer expert IDs, one Top-K row per token.
        topk_weights: Floating-point Router weights on the token device.
        group: EP participants entering this call in the same order.

    Returns:
        Router-weighted local output with the input token shape.

    Raises:
        ValueError: On any peer's incompatible routing or expert contract.
    """
    return fixed_slot_routed_forward(experts, hidden_states, RoutingResult(topk_indices, topk_weights), group)
