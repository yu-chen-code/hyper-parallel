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
"""Check reference attention after flattening its batch/head matmuls."""

import unittest

import torch

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    _reference_sparse_attention,
)


def _four_dimensional_reference(query, key_value, indices, sinks, scale):
    """Keep the original broadcast matmul equation as a numeric baseline."""
    batch_size, heads, sequence, _ = query.shape
    key_length = key_value.shape[2]
    valid = indices >= 0
    safe_indices = indices.clamp_min(0).long()
    allowed_counts = torch.zeros(batch_size, sequence, key_length, dtype=torch.int32)
    allowed_counts.scatter_add_(2, safe_indices, valid.to(torch.int32))
    logits = torch.matmul(query, key_value.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~(allowed_counts > 0).unsqueeze(1), float("-inf"))
    sink_logits = sinks.view(1, heads, 1, 1).expand(batch_size, -1, sequence, -1)
    probabilities = torch.cat((logits, sink_logits), dim=-1).softmax(dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities[..., :-1], key_value).transpose(1, 2)


class TestReferenceAttentionBatchMatmul(unittest.TestCase):
    """Flattening a broadcast head must preserve output and gradients."""

    def test_forward_and_gradients_with_shared_or_separate_keys(self):
        """Compare both accepted key-head layouts with sparse invalid slots."""
        indices = torch.tensor(
            [[[0, 1, -1], [1, 2, -1], [2, 3, -1], [3, 4, -1]]] * 2
        )
        for key_heads in (1, 3):
            with self.subTest(key_heads=key_heads):
                torch.manual_seed(47)
                query = torch.randn(2, 3, 4, 8, requires_grad=True)
                key_value = torch.randn(2, key_heads, 5, 8, requires_grad=True)
                sinks = torch.randn(3, requires_grad=True)
                actual = _reference_sparse_attention(
                    query, key_value, indices, sinks, 0.25
                )
                expected = _four_dimensional_reference(
                    query, key_value, indices, sinks, 0.25
                )
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                actual_grads = torch.autograd.grad(
                    actual.square().mean(), (query, key_value, sinks), retain_graph=True
                )
                expected_grads = torch.autograd.grad(
                    expected.square().mean(), (query, key_value, sinks)
                )
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, atol=1e-6, rtol=1e-6
                    )
