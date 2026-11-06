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
"""Tensor-backed launch/wait adapters for captured CP communication."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.compile import GraphCompiler
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.distributed.context_parallel import collectives


def _loss(model: torch.nn.Module, *, x: torch.Tensor) -> torch.Tensor:
    """A small traceable objective for compiler topology validation."""
    return model(x).sum()


class TestGraphCPCollectives(unittest.TestCase):
    """Mock only the functional transport; retain tensor layouts and autograd."""

    def test_allgather_wait_restores_layout_and_keeps_autograd(self) -> None:
        """The graph handle waits at the consumer and retains both replica contributions."""
        tensor = torch.randn(1, 3, 4, requires_grad=True)
        mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "cp")
        prefix = "hyper_parallel.distributed.context_parallel.collectives"
        with (
            patch(f"{prefix}._is_tracing_or_compiling", return_value=True),
            patch(f"{prefix}._functional_collectives.all_gather_single",
                  side_effect=lambda value, *unused: torch.cat((value, value))) as gather,
            patch(f"{prefix}._functional_collectives.wait_tensor", side_effect=lambda value: value) as wait,
            patch(f"{prefix}.dist") as eager,
        ):
            handle = collectives.async_cp_allgather_launch(tensor, 1, mesh)
            wait.assert_not_called()
            result = handle.wait()
            torch.testing.assert_close(result, torch.cat((tensor, tensor), dim=1), rtol=0, atol=0)
            result.sum().backward()
            torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 2), rtol=0, atol=0)
            self.assertEqual(gather.call_args.args[1:], (0, "cp"))
            wait.assert_called_once()
            self.assertEqual(eager.mock_calls, [])
            with self.assertRaisesRegex(RuntimeError, "waited twice"):
                handle.wait()

    def test_cp_ep_topology_is_supported(self) -> None:
        """CP may compose with EP after its collectives gain a graph-safe path."""
        context = SimpleNamespace(ep_size=2, cp_size=2, pp_size=1)
        compiler = GraphCompiler(torch.nn.Linear(4, 4), _loss, PassConfig(enable_overlap=False),
                                 mesh_context=context, device=torch.device("cpu"))
        compiler._validate_ep_topology()

    def test_ulysses_wait_restores_sequence_and_head_layout(self) -> None:
        """Functional A2A retains the same split/reconstruct contract and gradient edge."""
        tensor = torch.randn(1, 3, 4, requires_grad=True)
        mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "cp")
        prefix = "hyper_parallel.distributed.context_parallel.collectives"
        with (
            patch(f"{prefix}._is_tracing_or_compiling", return_value=True),
            patch(f"{prefix}._functional_collectives.all_to_all_single",
                  side_effect=lambda value, *unused: value) as exchange,
            patch(f"{prefix}._functional_collectives.wait_tensor", side_effect=lambda value: value) as wait,
            patch(f"{prefix}.dist") as eager,
        ):
            handle = collectives.async_ulysses_seq_to_head_launch(tensor, 1, -1, mesh)
            wait.assert_not_called()
            result = handle.wait()
            torch.testing.assert_close(result, torch.cat(tensor.chunk(2, dim=-1), dim=1), rtol=0, atol=0)
            result.sum().backward()
            torch.testing.assert_close(tensor.grad, torch.ones_like(tensor), rtol=0, atol=0)
            self.assertEqual(exchange.call_args.args[1:], (None, None, "cp"))
            wait.assert_called_once()
            self.assertEqual(eager.mock_calls, [])
