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
"""Traceable collective compatibility without changing eager dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.utils import communication


class TestDifferentiableCollectives(unittest.TestCase):
    """Support both functional API spellings and preserve payload gradients."""

    def test_graph_all_reduce_matches_legacy_backward(self):
        """A non-autograd functional kernel still reduces the backward payload."""
        for op, expected_op, scale in (("sum", "sum", 2), ("mean", "avg", 1)):
            with self.subTest(op=op):
                inputs = torch.randn(2, 3, requires_grad=True)
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication, "_functional_collectives") as backend:
                    backend.all_reduce.side_effect = lambda value, *_args, factor=scale: value.detach() * factor
                    backend.wait_tensor.side_effect = lambda value: value
                    actual = communication.differentiable_all_reduce(inputs, op, "tp")
                    actual.sum().backward()
                self.assertEqual(backend.all_reduce.call_count, 2)
                self.assertEqual(backend.wait_tensor.call_count, 2)
                for call in backend.all_reduce.call_args_list:
                    self.assertTrue(call.args[0].is_contiguous())
                    self.assertEqual(call.args[1:], (expected_op, "tp"))
                torch.testing.assert_close(actual, inputs * scale)
                torch.testing.assert_close(inputs.grad, torch.full_like(inputs, scale))

    def test_graph_all_gather_keeps_nonzero_dim_and_autograd(self):
        """Both API generations concatenate the requested axis with gradients."""
        for name in ("all_gather_single_autograd", "all_gather_tensor_autograd"):
            for axis in (2, -1, -2):
                with self.subTest(name=name, axis=axis):
                    inputs = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
                    collective = Mock(side_effect=lambda value, *_args: torch.cat((value, value + 100)))
                    backend = SimpleNamespace(**{name: collective})
                    with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                            patch.object(communication, "_functional_collectives", backend), \
                            patch.object(communication.dist, "get_world_size", return_value=2):
                        actual = communication.differentiable_all_gather_concat(inputs, "tp", 2, axis)
                        upstream = torch.arange(actual.numel(), dtype=inputs.dtype).reshape(actual.shape)
                        actual.backward(upstream)
                    collective.assert_called_once_with(inputs, 0, "tp")
                    torch.testing.assert_close(actual, torch.cat((inputs, inputs + 100), dim=axis), rtol=0, atol=0)
                    left, right = upstream.chunk(2, dim=axis)
                    torch.testing.assert_close(inputs.grad, left + right, rtol=0, atol=0)

    def test_graph_reduce_scatter_keeps_average_and_autograd(self):
        """Average normalization follows the differentiable SUM collective."""
        for name in ("reduce_scatter_single_autograd", "reduce_scatter_tensor_autograd"):
            with self.subTest(name=name):
                inputs = torch.arange(8.0).reshape(2, 4).requires_grad_()
                collective = Mock(side_effect=lambda value, _op, axis, _group: value.chunk(2, axis)[0] * 2)
                backend = SimpleNamespace(**{name: collective})
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication, "_functional_collectives", backend):
                    actual = communication.differentiable_reduce_scatter(inputs, 2, 1, "avg", "tp")
                    actual.sum().backward()
                collective.assert_called_once_with(inputs, "sum", 1, "tp")
                torch.testing.assert_close(actual, inputs[:, :2])
                torch.testing.assert_close(inputs.grad, torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.]]))

    def test_graph_collective_backward_materializes_upstream_gradient(self):
        """Non-contiguous upstream gradients are contiguous at collective backward."""
        for gather in (True, False):
            with self.subTest(gather=gather):
                inputs = torch.randn(2 if gather else 4, 3, requires_grad=True)
                payload = torch.cat((inputs, inputs)) if gather else inputs[:2]
                received = []
                payload.register_hook(lambda gradient, target=received: target.append(gradient.is_contiguous()))
                name = "all_gather_tensor_autograd" if gather else "reduce_scatter_tensor_autograd"
                backend = SimpleNamespace(**{name: Mock(return_value=payload)})
                upstream = torch.arange(payload.numel(), dtype=inputs.dtype).reshape(3, -1).t()
                self.assertFalse(upstream.is_contiguous())
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication, "_functional_collectives", backend), \
                        patch.object(communication.dist, "get_world_size", return_value=2):
                    if gather:
                        output = communication.differentiable_all_gather_concat(inputs, "tp", 2, 0)
                    else:
                        output = communication.differentiable_reduce_scatter(inputs, 2, 0, "sum", "tp")
                    output.backward(upstream)
                self.assertEqual(received, [True])
                expected = (
                    upstream[:2] + upstream[2:]
                    if gather else torch.cat((upstream, torch.zeros_like(upstream)))
                )
                torch.testing.assert_close(inputs.grad, expected)

    def test_eager_collectives_keep_legacy_dispatch(self):
        """The compatibility selection is absent from the eager path."""
        inputs = torch.ones(2, 4)
        with patch.object(communication, "_is_tracing_or_compiling", return_value=False), \
                patch.object(communication, "_functional_collectives") as backend, \
                patch.object(communication.dist_func, "all_gather", return_value=(inputs, inputs)) as gather, \
                patch.object(communication.dist_func, "reduce_scatter", return_value=inputs[:, :2] * 2) as scatter, \
                patch.object(communication.dist_func, "all_reduce", return_value=inputs * 2) as reduce:
            actual_gather = communication.differentiable_all_gather_concat(inputs, "tp", 2, 1)
            actual_scatter = communication.differentiable_reduce_scatter(inputs, 2, 1, "avg", "tp")
            actual_reduce = communication.differentiable_all_reduce(inputs, "sum", "tp")
        gather.assert_called_once_with(inputs, group="tp")
        self.assertEqual(scatter.call_count, 1)
        reduce.assert_called_once_with(inputs, op=torch.distributed.ReduceOp.SUM, group="tp")
        self.assertEqual(backend.mock_calls, [])
        torch.testing.assert_close(actual_gather, torch.cat((inputs, inputs), dim=1))
        torch.testing.assert_close(actual_scatter, inputs[:, :2])
        torch.testing.assert_close(actual_reduce, inputs * 2)
