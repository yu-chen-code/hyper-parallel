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
"""Prepared document compression metadata across changing packed boundaries."""

import copy
import unittest

import torch
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.distributed.context_parallel.compressed_document import document_compression_plan

_BOUNDARIES = ((0, 12, 24), (0, 7, 16, 24), (0, 5, 10, 17, 24), (0, 3, 8, 15, 24))


class _CompressionModel(nn.Module):
    """Apply trainable key and pooling projections with document-local geometry."""

    def __init__(self) -> None:
        """Create the two projections used by the original pooling formula."""
        super().__init__()
        self.key = nn.Linear(8, 8)
        self.gate = nn.Linear(8, 8)

    def forward(self, hidden: torch.Tensor, plan) -> torch.Tensor:
        """Pool complete blocks and retain their original rotary positions."""
        bank = plan.pool(self.key(hidden), self.gate(hidden), None)
        cos, sin = plan.positions(hidden.cos(), hidden.sin(), None)
        return bank * cos + sin


def _loss(model: nn.Module, *, hidden: torch.Tensor, plan) -> torch.Tensor:
    """Exercise both the projection and metadata paths in backward."""
    return model(hidden, plan).square().mean()


class TestPreparedDocumentCompression(unittest.TestCase):
    """Separate CPU document planning from tensor-only graph replay."""

    def test_prepared_pool_positions_and_gradients_match_eager_exactly(self) -> None:
        """Use the same rows, block starts, dtype and gradients as the eager plan."""
        for dtype in (torch.float32, torch.bfloat16):
            for boundaries in _BOUNDARIES:
                with self.subTest(dtype=dtype, boundaries=boundaries):
                    plan = document_compression_plan(boundaries, 4, 24, 1, 0)
                    prepared = plan.prepare(torch.device("cpu"))
                    key = torch.randn(1, 24, 8, dtype=dtype, requires_grad=True)
                    gate = torch.randn_like(key, requires_grad=True)
                    expected = plan.pool(key, gate, None)
                    actual = prepared.pool(key, gate)
                    self.assertTrue(torch.equal(actual, expected))
                    expected_grads = torch.autograd.grad(expected.sum(), (key, gate))
                    actual_grads = torch.autograd.grad(actual.sum(), (key, gate))
                    for left, right in zip(actual_grads, expected_grads):
                        self.assertTrue(torch.equal(left, right))
                    for left, right in zip(prepared.positions(key, gate), plan.positions(key, gate, None)):
                        self.assertTrue(torch.equal(left, right))

    def test_dynamic_block_counts_and_boundary_values_reuse_joint_graph(self) -> None:
        """Fixed S may change document count, incomplete tails and complete-block count."""
        torch.manual_seed(62)
        reference = _CompressionModel()
        candidate = copy.deepcopy(reference)
        compiler = GraphCompiler(candidate, _loss, PassConfig(fsdp_enabled=False), device=torch.device("cpu"),
                                 dynamic_arg_dims={"plan.rows": [0], "plan.block_positions": [0]})
        graph = None
        for boundaries in _BOUNDARIES:
            hidden = torch.randn(1, 24, 8)
            plan = document_compression_plan(boundaries, 4, 24, 1, 0)
            expected = _loss(reference, hidden=hidden, plan=plan)
            expected.backward()
            actual, _ = compiler.forward_backward(hidden=hidden, plan=plan.prepare(hidden.device))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            for left, right in zip(candidate.parameters(), reference.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
            if graph is None:
                graph = compiler._joint_graph
            self.assertIs(compiler._joint_graph, graph)

    def test_cp_routing_requires_a_separate_tensor_contract(self) -> None:
        """Do not silently discard boundary exchanges or CP bank ownership."""
        plan = document_compression_plan((0, 7, 16, 24), 4, 12, 2, 0)
        with self.assertRaisesRegex(ValueError, "requires CP=1"):
            plan.prepare(torch.device("cpu"))
