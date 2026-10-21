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
"""Keep broadcast matmul adaptation scoped to EP graph tracing."""

import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel.compile.expert_parallel import graph_ep_trace


def _joint(query, key):
    """Expose output and both broadcast operand gradients in one FX trace."""
    output = torch.matmul(query, key.transpose(-1, -2))
    return output, *torch.autograd.grad(output.square().mean(), (query, key))


class TestEPCaptureMatmul(unittest.TestCase):
    """The graph adapter must preserve math and leave eager execution untouched."""

    def test_joint_broadcast_gradient_and_scope_restore(self):
        """Trace explicit bmm batches and restore the original matmul function."""
        torch.manual_seed(19)
        query = torch.randn(2, 3, 4, 8, requires_grad=True)
        key = torch.randn(2, 1, 5, 8, requires_grad=True)
        original = torch.matmul
        expected = _joint(query, key)
        with graph_ep_trace(True):
            graph = make_fx(_joint)(query, key)
        self.assertIs(torch.matmul, original)
        actual = graph(query, key)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
        self.assertTrue(any("bmm" in str(node.target) for node in graph.graph.nodes))
        for left, right in zip(_joint(query, key), expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_disabled_capture_and_exception_restore(self):
        """Disabled scopes and exceptional exits retain the eager operation."""
        value = torch.randn(2, 3)
        expected = torch.matmul(value, value.t())
        with graph_ep_trace(False):
            torch.testing.assert_close(torch.matmul(value, value.t()), expected, atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with graph_ep_trace(True):
                raise RuntimeError("stop")
        torch.testing.assert_close(torch.matmul(value, value.t()), expected, atol=0, rtol=0)
