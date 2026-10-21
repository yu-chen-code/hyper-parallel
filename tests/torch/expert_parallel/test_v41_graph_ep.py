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
"""Framework-free launchers for real one-layer V4.1 graph EP acceptance."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).resolve().parent / "_test_v41_graph_ep.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_v41_one_layer_ep():
    """Run two BF16 V4.1 updates with two EP ranks."""
    torchrun_case(_WORKER, "test_v41_one_layer_ep", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_v41_one_layer_fsdp_ep():
    """Run two BF16 V4.1 updates with four FSDP+EP ranks."""
    torchrun_case(_WORKER, "test_v41_one_layer_fsdp_ep", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_v41_one_layer_ep_fp32():
    """Run ten FP32 V4.1 updates with two EP ranks."""
    torchrun_case(_WORKER, "test_v41_one_layer_ep_fp32", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_v41_one_layer_fsdp_ep_fp32():
    """Run ten FP32 V4.1 updates with four FSDP+EP ranks."""
    torchrun_case(_WORKER, "test_v41_one_layer_fsdp_ep_fp32", num_proc=4)
