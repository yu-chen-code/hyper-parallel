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
"""Expert-major rows shared by eager and captured expert execution."""

from dataclasses import dataclass
from typing import Protocol, Union

import torch


@dataclass(frozen=True)
class ExpertBatch:
    """Rows grouped by local expert with exclusive end offsets.

    ``offsets`` contains monotonically increasing exclusive end positions,
    one per local expert; the last value equals ``tokens.shape[0]``. Eager
    routing uses a one-dimensional tensor with dynamic values. Graph routing
    uses a tuple of Python shape values, so tracing never reads a tensor on
    the host. Graph batches may include inactive padded rows; offsets express
    capacity, not token validity.

    Args:
        tokens: Expert-major rows with shape ``[rows, hidden]``.
        offsets: Exclusive end positions as a dynamic one-dimensional tensor
            or a static tuple of Python integers.
    """

    tokens: torch.Tensor
    offsets: Union[torch.Tensor, tuple[int, ...]]


@dataclass(frozen=True)
class RoutingResult:
    """Router-owned tensors, both shaped [tokens, top_k].

    Attributes:
        indices: Integer global expert IDs on the hidden-state device.
        weights: Floating-point weights on that device, cast by combine.
    """

    indices: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class DispatchMetadata:
    """Per-call combine state; no parameter tensors or capture objects.

    Attributes:
        input_shape: Original token shape including the hidden dimension.
        group_size: EP participant count.
        local_expert_count: Number of local logical experts.
        slots: Local token count times Top-K.
        split_sizes: Fixed rows exchanged per peer in rank order.
        routing: Router tensors needed for weighting and ID validity.
    """

    input_shape: tuple[int, ...]
    group_size: int
    local_expert_count: int
    slots: int
    split_sizes: tuple[int, ...]
    routing: RoutingResult


@dataclass(frozen=True)
class DispatchPlan:
    """Static preparation result for this invocation only.

    Attributes:
        metadata: Combine layout and Router tensors.
        signature: Communication fields compared by peer preflight. The common
            pipeline appends compute device type and expert output dtype.
    """

    metadata: DispatchMetadata
    signature: tuple


@dataclass(frozen=True)
class DispatchResult:
    """Dispatched inputs and the matching per-call combine state.

    Attributes:
        batch: Expert-major tokens with exclusive expert end offsets.
        metadata: Layout and Router weights for this batch's combine.
    """

    batch: ExpertBatch
    metadata: DispatchMetadata


class TokenDispatcher(Protocol):
    """Prepare without data communication, then dispatch and combine."""

    group: object
    supports_graph_capture: bool

    def prepare(self, hidden_states: torch.Tensor, routing: RoutingResult) -> DispatchPlan:
        """Build a local plan before the EP group agrees to communicate.

        Args:
            hidden_states: Local token tensor with trailing hidden dimension.
            routing: This call's Top-K expert IDs and weights.

        Returns:
            Static layout and Router tensors owned by the current call.

        Raises:
            ValueError: If the backend's local input contract is violated.
        """
        raise NotImplementedError

    def dispatch(self, hidden_states: torch.Tensor, plan: DispatchPlan) -> DispatchResult:
        """Exchange tokens and return contiguous local expert groups.

        Args:
            hidden_states: Same input used to prepare the plan.
            plan: Prepared communication layout accepted by participating peers.

        Returns:
            Expert-major batch and its corresponding combine metadata.
        """
        raise NotImplementedError

    def combine(self, expert_output: torch.Tensor, metadata: DispatchMetadata) -> torch.Tensor:
        """Return and aggregate Router-weighted expert outputs.

        Args:
            expert_output: Rows corresponding to the dispatched ExpertBatch.
            metadata: Metadata returned by that invocation's dispatch.

        Returns:
            Local output with the original token layout.
        """
        raise NotImplementedError
