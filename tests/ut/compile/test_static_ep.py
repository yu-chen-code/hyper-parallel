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

"""Validate static collectives without changing native eager execution."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.compile.ep_collectives import ep_all_to_all


class TestStaticEPCollectives(unittest.TestCase):
    """Reject invalid fixed layouts before any token exchange."""

    def test_invalid_splits_do_not_communicate(self):
        """Check size, integral type, sign and row coverage."""
        with (patch('hyper_parallel.compile.ep_collectives.dist.is_initialized', return_value=True),
              patch('hyper_parallel.compile.ep_collectives.dist.get_world_size', return_value=2),
              patch('hyper_parallel.compile.ep_collectives.fc.all_to_all_single_autograd') as exchange):
            for send, receive in (([4], [2, 2]), ([True, 3], [2, 2]),
                                  ([-1, 5], [2, 2]), ([1, 2], [2, 2])):
                with self.subTest(send=send, receive=receive), self.assertRaises(ValueError):
                    ep_all_to_all(torch.ones(4, 8), send, receive, object())
            exchange.assert_not_called()

    def test_exchange_uses_differentiable_functional_collective(self):
        """Preserve unequal static splits and process group identity."""
        group = object()
        tokens = torch.randn(4, 8)
        with (patch('hyper_parallel.compile.ep_collectives.dist.is_initialized', return_value=True),
              patch('hyper_parallel.compile.ep_collectives.dist.get_world_size', return_value=2),
              patch('hyper_parallel.compile.ep_collectives.fc.all_to_all_single_autograd') as exchange):
            ep_all_to_all(tokens, [1, 3], [2, 5], group)
            exchange.assert_called_once_with(tokens, [2, 5], [1, 3], group)
