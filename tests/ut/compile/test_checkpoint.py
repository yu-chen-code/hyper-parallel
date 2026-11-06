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
"""Graph checkpoint payloads retain global ownership and discard physical padding."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from hyper_parallel.compile.checkpoint import GraphCheckpointAdapter
from hyper_parallel.core.distributed_checkpoint.metadata import CHUNK_INFO
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Shard


def _adapter(rank: int, source: tuple | None = None) -> GraphCheckpointAdapter:
    """Build a metadata-only two-rank storage owner without distributed initialization."""
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(3, 3))
    model.alias = model.weight
    model.scalar = torch.nn.Parameter(torch.tensor(2.0))
    model.register_buffer("counter", torch.tensor(4))
    mesh = SimpleNamespace(mesh_dim_names=("fsdp_shard",), get_local_rank=lambda axis: rank)
    compiler = SimpleNamespace(
        model=model, prepare_model=lambda: None,
        parameter_shapes={"weight": (5, 3), "alias": (5, 3), "scalar": ()},
        sharded_parameter_names={"weight"}, parameter_layouts={model.weight: SimpleNamespace(mesh=mesh)},
        source_layout_for_parameter=lambda name: source if name == "weight" else None,
    )
    return GraphCheckpointAdapter(compiler)


class TestGraphCheckpoint(TestCase):
    """Use the existing tensor chunk protocol for model, moments and main weights."""

    def test_padding_alias_and_optimizer_ownership_roundtrip(self):
        """A trailing shard stores two valid rows and reloads all physical tensors correctly."""
        adapter = _adapter(rank=1)
        state = adapter.compiler.model.state_dict()
        moment = torch.ones(3, 3)
        payload = {
            "model": state,
            "optimizer": {"state": {"weight": {"step": torch.tensor(5), "exp_avg": moment}},
                          "mixed_precision_optimizer": {"fp32_from_fp16_params": {"weight": moment.clone()}}},
            "extra_state": {"weight": torch.ones(17)},
        }
        checkpoint = adapter.prepare(payload)
        for value in (checkpoint["model"]["weight"], checkpoint["model"]["alias"],
                      checkpoint["optimizer"]["state"]["weight"]["exp_avg"],
                      checkpoint["optimizer"]["mixed_precision_optimizer"]["fp32_from_fp16_params"]["weight"]):
            info = getattr(value, CHUNK_INFO)
            self.assertEqual(info.global_shape, (5, 3))
            self.assertEqual(info.chunk.offsets, (3, 0))
            self.assertEqual(info.chunk.sizes, (2, 3))
            self.assertEqual(value.shape, (2, 3))
        self.assertFalse(hasattr(moment, CHUNK_INFO))
        self.assertEqual(adapter.compiler.model.weight.shape, (3, 3))
        restored = adapter.restore(checkpoint)
        torch.testing.assert_close(restored["model"]["weight"][:2], state["weight"][:2], rtol=0, atol=0)
        self.assertEqual(restored["model"]["weight"][2].count_nonzero(), 0)
        self.assertEqual(restored["optimizer"]["state"]["weight"]["step"], 5)
        self.assertEqual(restored["model"]["counter"], 4)
        self.assertIs(checkpoint["extra_state"], payload["extra_state"])
        self.assertEqual(getattr(checkpoint["model"], "_metadata"), getattr(state, "_metadata"))

    def test_flat_optimizer_and_invalid_state(self):
        """Flat FQN moments work; unsupported factorization or integer keys fail explicitly."""
        adapter = _adapter(rank=0)
        flat = adapter.prepare({"optimizer": {"state.weight.exp_avg": torch.ones(3, 3)}})
        self.assertEqual(getattr(flat["optimizer"]["state.weight.exp_avg"], CHUNK_INFO).chunk.offsets, (0, 0))
        for optimizer in ({"state": {0: {"exp_avg": torch.ones(3, 3)}}},
                          {"state": {"weight": {"exp_avg": torch.ones(3)}}}):
            with self.subTest(optimizer=optimizer), self.assertRaises(ValueError):
                adapter.prepare({"optimizer": optimizer})

    def test_source_shard_precedes_padded_fsdp_chunk(self):
        """FSDP's valid tail stays inside its source TP shard rather than the padded global shape."""
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.is_initialized", return_value=True), \
             patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=3):
            source_mesh = DeviceMesh("cpu", [1, 3], mesh_dim_names=("tp",), _init_backend=False)
        adapter = _adapter(rank=1, source=((Shard(0),), source_mesh))
        value = adapter.prepare({"model": adapter.compiler.model.state_dict()})["model"]["weight"]
        info = getattr(value, CHUNK_INFO)
        self.assertEqual(info.global_shape, (10, 3))
        self.assertEqual(info.chunk.offsets, (8, 0))
        self.assertEqual(info.chunk.sizes, (2, 3))
