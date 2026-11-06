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
"""Preserve unused-gradient slots without confusing them with tensor zeros."""

import copy
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.tracer import trace_model_graph


class _Model(nn.Module):
    """Keep a never-used parameter between two distinct trainable paths."""

    def __init__(self, use_all: bool = False) -> None:
        """Use zero as a real autograd input and optionally do the same for unused."""
        super().__init__()
        self.left = nn.Parameter(torch.randn(4, 4))
        self.unused = nn.Parameter(torch.ones(4, 4))
        self.zero = nn.Parameter(torch.ones(4, 4))
        self.right = nn.Parameter(torch.randn(4, 4))
        self.use_all = use_all

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Give zero a tensor gradient while unused has no loss dependency."""
        value = hidden @ self.left + hidden.square() @ self.right.tanh() + self.zero.sum() * 0
        return value + self.unused.sum() * 0 if self.use_all else value


def _loss(model: nn.Module, *, hidden: torch.Tensor) -> tuple:
    """Retain a trailing metric to check the gradient-output boundary."""
    loss = model(hidden).square().mean()
    return loss, {"mean": loss.detach()}


class TestUnusedGradients(unittest.TestCase):
    """Match eager optimizer skipping, output ordering and parallel admission."""

    def test_adamw_skips_unused_but_decays_used_zero_gradient(self) -> None:
        """Ten updates retain None and correct moments without shifting later grads."""
        torch.manual_seed(81)
        reference = _Model()
        candidate = copy.deepcopy(reference)
        initial = candidate.unused.detach().clone()
        compiler = GraphCompiler(candidate, _loss, PassConfig(fsdp_enabled=False), device=torch.device("cpu"))
        optimizers = [torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1, foreach=False)
                      for model in (reference, candidate)]
        for _ in range(10):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                hidden = torch.randn(2, 4)
                expected, _ = _loss(reference, hidden=hidden)
                expected.backward()
                actual, metrics = compiler.forward_backward(hidden=hidden)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(metrics["mean"], expected.detach())
                for eager, compiled in zip(reference.parameters(), candidate.parameters()):
                    if eager.grad is None:
                        self.assertIsNone(compiled.grad)
                    else:
                        torch.testing.assert_close(compiled.grad, eager.grad)
            for optimizer in optimizers:
                optimizer.step()
            for eager, compiled in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(compiled, eager)
        self.assertTrue(torch.equal(candidate.unused, initial))
        self.assertNotIn(candidate.unused, optimizers[1].state)
        self.assertIn(candidate.zero, optimizers[1].state)
        self.assertTrue(torch.all(candidate.zero < initial))
        for eager, compiled in zip(reference.parameters(), candidate.parameters()):
            for key, expected in optimizers[0].state.get(eager, {}).items():
                torch.testing.assert_close(optimizers[1].state[compiled][key], expected)

    def test_none_does_not_clear_an_earlier_microbatch_gradient(self) -> None:
        """A skipped output must leave an already accumulated gradient intact."""
        model = _Model()
        compiler = GraphCompiler(model, _loss, PassConfig(fsdp_enabled=False), device=torch.device("cpu"))
        previous = torch.full_like(model.unused, 2.)
        model.unused.grad = previous
        compiler.forward_backward(hidden=torch.randn(2, 4))
        self.assertIs(model.unused.grad, previous)
        self.assertTrue(torch.equal(model.unused.grad, torch.full_like(previous, 2.)))

    def test_fsdp_reduces_used_slots_without_compacting_unused(self) -> None:
        """The original parameter index survives None and the trailing metric."""
        model = _Model()
        joint = trace_model_graph(model, _loss, {"hidden": torch.randn(2, 4)})
        mocked = MagicMock()
        mocked.is_initialized.return_value = True
        mocked.get_world_size.return_value = 2
        mocked.get_rank.return_value = 0
        with (patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist", mocked),
              patch("hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group")):
            FSDPPass().run(joint.graph_module, PassConfig(fsdp_degree=2), model=model)
        outputs = next(node for node in joint.graph_module.graph.nodes if node.op == "output").args[0]
        self.assertEqual(len(outputs), 6)
        self.assertIsNone(outputs[2])
        self.assertEqual([outputs[index].meta["param_name"] for index in (1, 3, 4)], ["left", "zero", "right"])
        reductions = [node for node in joint.graph_module.graph.nodes
                      if "reduce_scatter_tensor" in str(node.target)]
        self.assertEqual(len(reductions), 3)

    def test_rank_dependent_usage_and_pipeline_unused_are_rejected(self) -> None:
        """All-used ranks also participate before any parallel lowering can run."""
        for use_all in (False, True):
            with self.subTest(use_all=use_all):
                model = _Model(use_all)
                compiler = GraphCompiler(model, _loss, PassConfig(fsdp_enabled=True))
                graph = trace_model_graph(model, _loss, {"hidden": torch.randn(2, 4)}).graph_module

                def gather(schemas, schema):
                    peer_mask = list(schema[1])
                    name, unused = peer_mask[1]
                    peer_mask[1] = (name, not unused)
                    schemas[:] = [schema, (schema[0], tuple(peer_mask))]

                with (patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=True),
                      patch("hyper_parallel.compile.compiler.dist.get_world_size", return_value=2),
                      patch("hyper_parallel.compile.compiler.dist.all_gather_object", side_effect=gather) as collect,
                      self.assertRaisesRegex(ValueError, "same parameter usage schema")):
                    compiler._validate_gradient_usage(graph)
                collect.assert_called_once()
        compiler = GraphCompiler(_Model(), _loss, PassConfig(fsdp_enabled=False, pp_enabled=True))
        with (patch("hyper_parallel.compile.compiler.PassPipeline.from_config") as pipeline,
              self.assertRaisesRegex(ValueError, "pipeline parallel does not support unused")):
            compiler.compile(hidden=torch.randn(2, 4))
        pipeline.assert_not_called()

    def test_rank_dependent_trainable_identity_is_rejected(self) -> None:
        """Identical state names and all-used flags may still belong to different parameters."""
        model, peer = _Model(use_all=True), _Model(use_all=True)
        model.left.requires_grad_(False)
        peer.right.requires_grad_(False)
        inputs = {"hidden": torch.randn(2, 4)}
        graph = trace_model_graph(model, _loss, inputs).graph_module
        peer_joint = trace_model_graph(peer, _loss, inputs)
        peer_schema = (tuple(peer_joint.state_fqns), tuple((name, False) for name in peer_joint.param_names))
        compiler = GraphCompiler(model, _loss, PassConfig(fsdp_enabled=True))

        def gather(schemas, schema):
            schemas[:] = [schema, peer_schema]

        with (patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=True),
              patch("hyper_parallel.compile.compiler.dist.get_world_size", return_value=2),
              patch("hyper_parallel.compile.compiler.dist.all_gather_object", side_effect=gather),
              self.assertRaisesRegex(ValueError, "same parameter usage schema")):
            compiler._validate_gradient_usage(graph)
