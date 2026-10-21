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

"""Opt-in graph V4.1 factory; native eager factories remain byte-identical."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import torch

from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.distributed.expert_parallel.experts import require_attrs
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    _configure_router_statistics, _deepseek_v41_unclamped_gate,
    _local_router_metadata, _resolve_router, _router,
)
from .ep_compute import bind_static_expert_forward
from .expert_parallel import routed_forward_with_router


def _routed_and_shared_forward(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None,
        ep_group: Any,
        router_token_mask: torch.Tensor | None = None,
        router_sequence_ids: torch.Tensor | None = None,
        router_num_sequences: int | None = None,
        sequence_partition_groups: tuple[Any, ...] = (),
        *,
        ep_size: int,
        fix_router: bool = False,
) -> torch.Tensor:
    """Dispatch routed experts and add the V4.1 shared expert branch."""
    def router_fn(target: Any, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route with statistics over the current token partitions."""
        return _router(
            target, states, image_mask, router_token_mask, router_sequence_ids,
            router_num_sequences, sequence_partition_groups=sequence_partition_groups,
        )

    routed = routed_forward_with_router(
        module, hidden_states,
        router_fn=_resolve_router(router_fn, ep_size=ep_size, fix_router=fix_router), ep_group=ep_group,
    )
    return routed + module.shared_experts(hidden_states)


@local_compute
def deepseek_v41_graph_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        use_grouped_gemm: bool = False,
        fix_router: bool = False,
) -> Callable:
    """Build explicit static V4.1 routing on a dedicated graph model.

    Sequence statistics reduce CP/TP token partitions; bias updates also reduce
    DP axes. Fixed routing changes dispatch indices after the gate, so enabled
    auxiliary statistics still describe learned decisions. Pure throughput runs
    can disable both router coefficients.

    Args:
        module: V4.1 MoE block exposing gate, experts, and shared_experts.
        mesh: Full mesh supplying data-parallel bias statistics groups.
        tp_mesh: Tensor-parallel mesh used to align router metadata and statistics.
        cp_mesh: Context-parallel mesh supplying token-partition statistics groups.
        ep_mesh: Expert-parallel mesh owning the routed experts.
        use_grouped_gemm: Must be false; grouped capture is outside this stage.
        fix_router: Must be false; static validation uses the learned Router.

    Returns:
        A graph-only local compute callback preserving the model call contract.

    Raises:
        ValueError: If mesh, routing, grouped compute or expert layout is unsupported.

    Note:
        Bind this factory only on a dedicated graph model. The native eager
        factory and variable-length dispatcher remain separate and unchanged.
    """
    if use_grouped_gemm or fix_router or tp_mesh is not None or cp_mesh is not None:
        raise ValueError("Static V4.1 EP supports learned routing without grouped GEMM or TP/CP")
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 EP")
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 EP forward requires an active ep_mesh")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 validation layers must use learned routing")
    if not isinstance(fix_router, bool):
        raise ValueError("fix_router must be a boolean")
    sequence_groups = _configure_router_statistics(module.gate, mesh, tp_mesh, cp_mesh)
    ep_group = ep_mesh.get_group("ep")
    ep_size = ep_mesh["ep"].size()
    apply_gate = (
        module.experts._apply_gate
        if module.experts.limit > 0 else _deepseek_v41_unclamped_gate
    )
    bind_static_expert_forward(module, ep_size, apply_gate=apply_gate)

    def auxiliary_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
            router_token_mask: torch.Tensor | None = None,
            router_sequence_ids: torch.Tensor | None = None,
            router_num_sequences: int | None = None,
    ) -> torch.Tensor:
        """Run text and image routing through the same dispatch/combine path.

        Args:
            module: MLP owning routed and shared experts.
            hidden_states: Token activations before expert dispatch.
            input_ids: Unused by learned routing.
            image_mask: Optional image-token mask.
            router_token_mask: Optional valid-token mask.
            router_sequence_ids: Optional packed logical sample IDs.
            router_num_sequences: Global sample count within this microbatch.
        """
        del input_ids
        return _routed_and_shared_forward(
            module, hidden_states, _local_router_metadata(image_mask, hidden_states, tp_mesh), ep_group,
            router_token_mask=_local_router_metadata(router_token_mask, hidden_states, tp_mesh),
            router_sequence_ids=_local_router_metadata(router_sequence_ids, hidden_states, tp_mesh),
            router_num_sequences=router_num_sequences, sequence_partition_groups=sequence_groups,
            ep_size=ep_size, fix_router=fix_router,
        )

    if "router_token_mask" in inspect.signature(module.forward).parameters:
        return auxiliary_compute_fn

    if "image_mask" not in inspect.signature(module.forward).parameters:
        def text_compute_fn(
                module: Any,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the text-only source contract without visual routing state.

            Args:
                module: Routed and shared expert module.
                hidden_states: Current decoder hidden rows.
                input_ids: Unused token IDs from the local compute contract.
            """
            del input_ids
            return _routed_and_shared_forward(
                module, hidden_states, None, ep_group, ep_size=ep_size, fix_router=fix_router,
            )

        return text_compute_fn

    def multimodal_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the multimodal source contract with optional visual routing state.

        Args:
            module: Routed and shared expert module.
            hidden_states: TP-local decoder hidden rows.
            input_ids: Unused token IDs from the local compute contract.
            image_mask: CP-local mask identifying image tokens.
        """
        del input_ids
        local_image_mask = _local_router_metadata(image_mask, hidden_states, tp_mesh)
        return _routed_and_shared_forward(
            module, hidden_states, local_image_mask, ep_group, ep_size=ep_size, fix_router=fix_router,
        )

    return multimodal_compute_fn
