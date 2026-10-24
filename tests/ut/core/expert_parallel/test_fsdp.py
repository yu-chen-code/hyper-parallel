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
"""Shared Expert FSDP placement and eager adapter integration."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.expert_parallel.fsdp import expert_shard_placement
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config


class TestExpertShardPlacement(unittest.TestCase):
    """Expert and inner sharding use the same rule in both execution modes."""

    def test_selects_divisible_axis_or_rejects(self):
        """Prefer whole experts, then inner axes, without padding experts."""
        for shape, degree, dim in (((4, 8, 6), 2, 0), ((1, 8, 6), 2, 1),
                                   ((3, 5, 8), 4, 2), ((1, 8, 6), 1, 0)):
            with self.subTest(shape=shape, degree=degree):
                self.assertEqual(expert_shard_placement(torch.empty(shape), shard_degree=degree).dim, dim)
        for shape, degree in (((), 2), ((1, 3, 5), 2), ((0, 3), 2), ((4, 8), 0)):
            with self.subTest(shape=shape, degree=degree), self.assertRaises(ValueError):
                expert_shard_placement(torch.empty(shape), shard_degree=degree)
        parameter = MagicMock(spec=DTensor)
        parameter.shape = (2, 8, 6)
        parameter.to_local.return_value = torch.empty(1, 8, 6)
        self.assertEqual(expert_shard_placement(parameter, shard_degree=2).dim, 1)

    def test_eager_expert_unit_installs_shared_placement(self):
        """The adapter passes the shared rule to fully_shard only for experts."""
        mesh = MagicMock()
        mesh.__getitem__.return_value.size.return_value = 2
        manager = FSDP2Manager(FSDP2Config(), SimpleNamespace(fsdp_moe_mesh=mesh))
        unit = SimpleNamespace(fqn="experts", module=torch.nn.Linear(4, 4))
        with (
            patch("hyper_parallel.distributed._builder.fsdp_adapter._build_managed_source_shard_info"),
            patch("hyper_parallel.distributed._builder.fsdp_adapter._source_infos_for_fully_shard"),
            patch("hyper_parallel.distributed._builder.fsdp_adapter.fully_shard") as wrap,
            patch.object(manager, "_uses_expert_mesh", return_value=True),
            patch.object(manager, "_build_fsdp_actual_mesh", return_value=mesh),
            patch.object(manager, "_build_fully_shard_kwargs", return_value=({}, {})),
            patch.object(manager, "_build_managed_replicate_params", return_value=None),
            patch.object(manager, "_configure_source_layout_gradient_scaling", return_value=False),
        ):
            manager._parallelize_child_units([unit], {}, {}, None, {})
        placement = wrap.call_args.kwargs["shard_placement_fn"]
        self.assertIs(placement.func, expert_shard_placement)
        self.assertEqual(placement(torch.empty(1, 8, 6)).dim, 1)
