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
"""Precision policy preserves storage and the complete optimizer update chain."""

from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig


def _loss(model, *, x):
    """Keep loss accumulation in FP32 while linear computation is BF16."""
    return model(x).float().square().mean()


def _compiler(model, **policy):
    """Use the same YAML policy resolver as TextTrainer."""
    config = SimpleNamespace(fsdp_config=FSDP2Config(wrapper_class=["_Block"],
                                                   mix_precision=FSDP2MixedPrecisionConfig(**policy)))
    return GraphCompiler(model, _loss, PassConfig(fsdp_enabled=False), trainer_config=config,
                         device=torch.device("cpu"))


class _Block(nn.Linear):
    """Explicitly selected FSDP child unit."""


class _Float32(nn.Module):
    """Model-owned explicit FP32 region, as used by routers and normalization."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Convert the activation before the FP32 projection."""
        return value.float()


class TestGraphMixedPrecision(TestCase):
    """The default input cast and per-parameter precision are graph operations."""

    def test_fp32_storage_bf16_compute_matches_ten_adamw_updates(self):
        """Compare losses, gradients, weights and moments against explicit eager casts."""
        torch.manual_seed(17)
        model = nn.Sequential(_Block(4, 4))
        reference = deepcopy(model)
        compiler = _compiler(model, param_dtype="bfloat16", reduce_dtype="float32")
        compiler.prepare_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
        for step in range(10):
            x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 16 + step / 64
            computation = deepcopy(reference).bfloat16()
            expected = computation(x.bfloat16())
            expected_loss = expected.float().square().mean()
            expected_loss.backward()
            for master, parameter in zip(reference.parameters(), computation.parameters()):
                master.grad = parameter.grad.float()
            loss, _ = compiler.forward_backward(x=x)
            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            for actual, expected_parameter in zip(model.parameters(), reference.parameters()):
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual.grad, expected_parameter.grad, rtol=0, atol=0)
            optimizer.step()
            expected_optimizer.step()
            for actual, expected_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected_parameter, rtol=0, atol=0)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(optimizer.state[actual][key],
                                               expected_optimizer.state[expected_parameter][key], rtol=0, atol=0)
            optimizer.zero_grad()
            expected_optimizer.zero_grad()
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model._forward_hooks)
        self.assertTrue(any(node.target == torch.ops.aten._to_copy.default
                            for node in compiler._joint_graph.graph_module.graph.nodes))

    def test_custom_parameter_dtype_and_module_output_cast(self):
        """Pattern overrides preserve an explicit FP32 compute region."""
        model = nn.Sequential(_Block(4, 4), _Float32(), nn.Linear(4, 4))
        compiler = _compiler(model, param_dtype="bfloat16", reduce_dtype="float32", output_dtype="float32",
                             custom_params={"2.*": {"param_dtype": "float32"}})
        compiler.prepare_model()
        precision = compiler._mixed_precision
        self.assertEqual(precision.parameter_dtypes["0.weight"], (torch.bfloat16, torch.float32))
        self.assertEqual(precision.parameter_dtypes["2.weight"], (torch.float32, torch.float32))
        state = dict(model.named_parameters())
        cast = precision.cast_state(state)
        self.assertEqual(cast["0.weight"].dtype, torch.bfloat16)
        self.assertIs(cast["2.weight"], model[2].weight)
        with precision.module_casts():
            # Module output casts use the same floating-leaf helper as eager.
            result = torch.func.functional_call(model, cast, (torch.ones(2, 4),))
        self.assertEqual(result.dtype, torch.float32)

    def test_fp32_main_gradient_keeps_reduction_precision(self):
        """A BF16 model can accumulate FP32 gradients without assigning them to .grad."""
        model = nn.Sequential(_Block(4, 4, dtype=torch.bfloat16))
        compiler = _compiler(model, param_dtype="bfloat16", reduce_dtype="float32")
        compiler._fp32_main_params = True
        compiler.prepare_model()
        compiler.forward_backward(x=torch.ones(2, 4))
        first = [parameter.main_grad.clone() for parameter in model.parameters()]
        compiler.forward_backward(x=torch.ones(2, 4))
        for parameter, gradient in zip(model.parameters(), first):
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.main_grad.dtype, torch.float32)
            torch.testing.assert_close(parameter.main_grad, gradient * 2, rtol=0, atol=0)
