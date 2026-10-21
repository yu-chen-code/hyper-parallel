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

"""Explicit static EP routing and scoped capture records; no eager interception."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
from torch.overrides import TorchFunctionMode

from hyper_parallel.distributed._builder.forward_rewriter import _install_bound_forward

from .ep_collectives import ep_all_to_all
from .ep_contract import DispatchPlan, RoutingResult
from .ep_dispatch import FixedSlotTokenDispatcher, routing_preflight
from .ep_compute import _expert_compute_dtype, _static_expert_forward, validate_expert_layout

__all__ = ["ep_all_to_all", "routed_forward", "routed_forward_with_router", "graph_ep_trace"]

_GRAPH_EP_CAPTURE = ContextVar("graph_ep_capture", default=None)


class _BroadcastMatmulCaptureMode(TorchFunctionMode):
    """Flatten broadcast batch matmuls only while joint EP graphs are traced.

    Stock torch 2.9.1 can collapse a broadcast key gradient during joint FX
    capture. Explicit bmm batches preserve that gradient layout while eager
    attention continues using its unchanged model implementation.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Rewrite the four-dimensional broadcast case and delegate other calls."""
        del types
        kwargs = kwargs or {}
        if func is torch.matmul and len(args) == 2 and not kwargs:
            left, right = args
            if (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                    and left.ndim == right.ndim == 4 and left.shape[:-2] != right.shape[:-2]):
                batch = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
                rows, inner, columns = left.shape[-2], left.shape[-1], right.shape[-1]
                left = left.expand(*batch, rows, inner).reshape(-1, rows, inner)
                right = right.expand(*batch, inner, columns).reshape(-1, inner, columns)
                return torch.bmm(left, right).reshape(*batch, rows, columns)
        return func(*args, **kwargs)


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


@contextmanager
def graph_ep_trace(enabled: bool, model: Optional[torch.nn.Module] = None) -> Iterator[GraphEPCaptureContext]:
    """Record explicit graph EP calls and restore the capture scope on failure.

    Args:
        enabled: Record explicit static EP calls when true; eager entry points are untouched.
        model: Functional model used to resolve exact expert parameter names.

    Yields:
        Capture record consumed by GraphCompiler; never redirects native eager calls.
    """
    context = GraphEPCaptureContext(model)
    if not enabled:
        yield context
        return
    token = _GRAPH_EP_CAPTURE.set(context)
    try:
        with _BroadcastMatmulCaptureMode():
            yield context
    finally:
        _GRAPH_EP_CAPTURE.reset(token)


def execute_routed_experts(experts: Any, hidden: torch.Tensor, router: Callable,
                           dispatcher: FixedSlotTokenDispatcher, group: Any) -> torch.Tensor:
    """Execute the dedicated static route after coordinated preparation.

    Args:
        experts: Bound holder called through its module/FSDP hooks.
        hidden: Tokens with the static shape agreed by EP participants.
        router: Callback producing this invocation's RoutingResult.
        dispatcher: Fixed-slot layout used only by this explicit static invocation.
        group: EP participants, identical to the bound dispatcher's group.

    Returns:
        Router-weighted local output with the input token shape.

    Raises:
        ValueError: If any participant's Router or expert preparation fails,
            after the group has exchanged its validation status.
    """
    capture = _GRAPH_EP_CAPTURE.get()

    def prepare() -> DispatchPlan:
        """Prepare Router and logical expert contracts before peer agreement."""
        if dispatcher.group is not group:
            raise ValueError("EP dispatcher group differs from the bound group")
        if capture is not None and not dispatcher.supports_graph_capture:
            raise ValueError("Selected EP dispatcher does not support graph capture")
        routing = router()
        plan = dispatcher.prepare(hidden, routing)
        if getattr(experts, "num_experts", plan.signature[1]) != plan.signature[1]:
            raise ValueError("Graph EP expert count conflicts with the eager binder's local_expert_count")
        weights = validate_expert_layout(experts, hidden)
        output_dtype = _expert_compute_dtype(experts, hidden, weights)
        if capture is not None:
            capture.record_weights(weights)
        return replace(plan, signature=(*plan.signature, hidden.device.type, str(output_dtype)))

    if capture is not None:
        plan = capture.preflight(group, prepare)
    else:
        plan = routing_preflight(group, prepare)
    result = dispatcher.dispatch(hidden, plan)
    output = experts(result.batch.tokens, offsets=result.batch.offsets)
    return dispatcher.combine(output, result.metadata)


def routed_forward_with_router(module: Any, hidden_states: torch.Tensor, *,
                               router_fn: Callable, ep_group: dist.ProcessGroup) -> torch.Tensor:
    """Run an explicitly bound graph MoE route, including Router preflight.

    Args:
        module: Dedicated graph MoE with statically bound expert-major forward.
        hidden_states: Static local token tensor with a trailing hidden axis.
        router_fn: Model-owned callback returning Top-K indices and weights.
        ep_group: All participants entering this route in the same order.

    Returns:
        Routed output; the model adapter composes shared experts separately.

    Raises:
        ValueError: If any participant's Router or layout preparation fails.
    """
    dispatcher = FixedSlotTokenDispatcher(ep_group, getattr(module.experts, "local_expert_count", 0))

    def router() -> RoutingResult:
        """Keep model-owned Router exceptions inside coordinated preparation."""
        return RoutingResult(*router_fn(module, hidden_states))

    return execute_routed_experts(module.experts, hidden_states, router, dispatcher, ep_group)


def routed_forward(experts: torch.nn.Module, hidden_states: torch.Tensor,
                   topk_indices: torch.Tensor, topk_weights: torch.Tensor,
                   group: dist.ProcessGroup) -> torch.Tensor:
    """Execute an explicit static route with already computed Router tensors.

    Args:
        experts: Dedicated static holder with local_expert_count and SwiGLU weights.
        hidden_states: Local floating-point tokens with static trailing width.
        topk_indices: Integer global expert IDs, shaped [tokens, top_k].
        topk_weights: Floating-point Router weights on the hidden-state device.
        group: EP participants entering this invocation in the same order.

    Returns:
        Router-weighted output with the input shape and expert output dtype.

    Raises:
        ValueError: If any participant's preparation or signature is invalid.

    Note:
        This explicit API installs static compute on the supplied holder.
        It is never selected by a native eager factory or GraphCompiler.
    """
    _install_bound_forward(experts, _static_expert_forward)
    dispatcher = FixedSlotTokenDispatcher(group, getattr(experts, "local_expert_count", 0))
    routing = RoutingResult(topk_indices, topk_weights)
    return execute_routed_experts(experts, hidden_states, lambda: routing, dispatcher, group)
