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
"""Launch two-rank graph EP correctness tests without importing Torch here."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).resolve().parent / "_test_graph_ep.py")




@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_gloo_equal():
    """Compare graph and eager EP across two Gloo ranks."""
    torchrun_case(_WORKER, "test_graph_ep_gloo_equal", num_proc=2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_gloo_agrees_on_local_validation_failure():
    """Share a single rank's local validation error across the EP group."""
    torchrun_case(_WORKER, "test_graph_ep_gloo_agrees_on_local_validation_failure",
                  num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_equal():
    """Compare graph and eager EP across two HCCL ranks with equal splits."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_equal", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_uneven():
    """Compare graph and eager EP across two HCCL ranks with uneven splits."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_uneven", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_zero():
    """Compare graph and eager EP when a peer receives zero rows."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_zero", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_routed():
    """Compare eager and graph Top-K routing through a shared forward."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_routed", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_routed_bf16_fp32_router():
    """Compare BF16 eager and graph routing with FP32 Top-K weights."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_routed_bf16_fp32_router", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_rejects_unequal_local_tokens():
    """Require equal fixed capacities on both EP ranks at graph capture."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_rejects_unequal_local_tokens", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_rejects_hidden_width_mismatch():
    """Reject unequal hidden widths on both HCCL ranks at graph capture."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_rejects_hidden_width_mismatch", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_rejects_split_overlap():
    """Reject the eager split dispatch during two-rank graph capture."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_rejects_split_overlap", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_agrees_on_split_dispatch_mismatch():
    """Report a single rank's split-dispatch configuration error on both ranks."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_agrees_on_split_dispatch_mismatch", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_agrees_on_preparation_failures():
    """Agree on one rank's Router or expert-weight failure before data exchange."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_agrees_on_preparation_failures", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_agrees_on_tensor_contract_failures():
    """Reject unilateral dtype/device mismatches before any data exchange."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_agrees_on_tensor_contract_failures", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_agrees_on_local_capture_failure():
    """Share one rank's local failure during real HCCL graph capture."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_agrees_on_local_capture_failure",
                  num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_fsdp_ep_hccl_routed():
    """Compare graph FSDP+EP with four-rank eager reference training."""
    torchrun_case(_WORKER, "test_graph_fsdp_ep_hccl_routed", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_fsdp_ep_hccl_rejects_unsupported_meshes():
    """Reject unsupported replica and TP mesh axes on four ranks."""
    torchrun_case(_WORKER, "test_graph_fsdp_ep_hccl_rejects_unsupported_meshes",
                  num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_fsdp_ep_hccl_rejects_nondivisible_experts():
    """Reject EDP shard degree four with only two local experts."""
    torchrun_case(_WORKER, "test_graph_fsdp_ep_hccl_rejects_nondivisible_experts",
                  num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_invalid_replay_is_nonfinite():
    """Verify the captured invalid-ID path signals failure on both ranks."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_invalid_replay_is_nonfinite", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_graph_ep_hccl_independent_subgroup_steps():
    """Run independent EP subgroups on different schedules without WORLD sync."""
    torchrun_case(_WORKER, "test_graph_ep_hccl_independent_subgroup_steps", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_explicit_static_route_against_native_dynamic():
    """Compare eager and graph Top-K routing through a shared forward."""
    torchrun_case(_WORKER, "test_explicit_static_route_against_native_dynamic", num_proc=2)
