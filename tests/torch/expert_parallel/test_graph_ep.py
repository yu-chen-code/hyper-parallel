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
def test_explicit_static_route_against_native_dynamic():
    """Compare eager and graph Top-K routing through a shared forward."""
    torchrun_case(_WORKER, "test_explicit_static_route_against_native_dynamic", num_proc=2)
