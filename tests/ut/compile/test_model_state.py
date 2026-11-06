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
"""Joint-graph replay of model-owned router buffers and shared attention state."""

import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hyper_parallel.compile import GraphCompiler
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.components.modules.moe_utils import MoEMonitorCallback
from hyper_parallel.components.modules.shared_compressed_dsa_attention import SharedCompressedAttentionState
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter


class _RouterBlock(nn.Module):
    """Keep the real V4.1 router inside a small checkpointable block."""

    def __init__(self) -> None:
        """Initialize a small projection and the trainable V4.1 router."""
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.router = DeepseekV41TopKRouter(SimpleNamespace(
            hidden_size=8, num_local_experts=4, num_experts_per_tok=2,
            scoring_func="sigmoid", routed_scaling_factor=1.0,
            router_aux_loss_coef=0.01, router_bias_update_rate=0.005,
        ))
        with torch.no_grad():
            self.router.weight.uniform_(-0.4, 0.4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Consume routing weights so backward records the expert loads."""
        hidden = self.linear(hidden).tanh()
        _, weights, indices = self.router(hidden)
        scale = (weights * (indices + 1)).sum(-1).view(*hidden.shape[:-1], 1)
        return hidden * scale


class _RouterModel(nn.Module):
    """Expose the ordinary HF block marker for the compiler's AC discovery."""

    def __init__(self) -> None:
        """Register the block container and a prediction head."""
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([_RouterBlock()])
        self.head = nn.Linear(8, 8)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the checkpointable router block and its prediction head."""
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(hidden)


class _SharedBlock(nn.Module):
    """Publish one differentiable KV tensor or consume the shared producer."""

    def __init__(self, publish: bool) -> None:
        """Configure this layer's producer or consumer role."""
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.publish = publish

    def forward(self, hidden: torch.Tensor, state: SharedCompressedAttentionState) -> torch.Tensor:
        """Keep all consumers connected to the same producer's autograd edge."""
        hidden = self.linear(hidden).tanh()
        if self.publish:
            state.publish_compressed_kv(0, hidden)
            return hidden
        return hidden + state.require_compressed_kv(0, 1).sin()


class _SharedModel(nn.Module):
    """Recreate the V4.1 shared-state lifetime on every model forward."""

    def __init__(self) -> None:
        """Create a producer and its two consumers."""
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([_SharedBlock(True), _SharedBlock(False), _SharedBlock(False)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Publish once and run two consumers without retaining a prior batch."""
        state = SharedCompressedAttentionState()
        for layer in self.layers:
            hidden = layer(hidden, state)
        return hidden


def _train_fn(model: nn.Module, *, x: torch.Tensor, y: torch.Tensor) -> tuple:
    """Scale task and auxiliary gradients consistently across two microbatches."""
    loss = (model(x) - y).square().mean()
    bind_aux_loss_scale(loss)
    metrics = {"foundation_loss": loss.detach()}
    routers = [module for module in model.modules() if isinstance(module, DeepseekV41TopKRouter)]
    if routers:
        metrics["router_aux_loss"] = torch.stack([router.last_aux_loss for router in routers]).mean()
    return loss / 2, metrics


def _check_router_microstep(compiler: GraphCompiler, reference: _RouterModel,
                            x: torch.Tensor, y: torch.Tensor) -> None:
    """Compare task loss, auxiliary metric and live counters after one backward."""
    with aux_loss_scale_context():
        expected_loss, expected_metrics = _train_fn(reference, x=x, y=y)
        expected_loss.backward()
    with aux_loss_scale_context():
        loss, metrics = compiler.forward_backward(x=x, y=y)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metrics, expected_metrics, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(compiler.model.layers[0].router.tokens_per_expert,
                               reference.layers[0].router.tokens_per_expert, rtol=0, atol=0)


class TestModelStateReplay(unittest.TestCase):
    """Model-owned state must retain eager semantics across graph replays."""

    def test_router_buffers_auxiliary_loss_and_optimizer_steps(self) -> None:
        """Count each microbatch once and expose new bias/weights on the next replay."""
        for mode in ("off", "full", "selective"):
            with self.subTest(ac_mode=mode):
                torch.manual_seed(41)
                reference = _RouterModel()
                model = copy.deepcopy(reference)
                compiler = GraphCompiler(model, _train_fn, PassConfig(fsdp_enabled=False, ac_mode=mode),
                                         device=torch.device("cpu"))
                ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
                ref_monitor = MoEMonitorCallback(reference)
                monitor = MoEMonitorCallback(model)
                for _ in range(3):
                    for _ in range(2):
                        x, y = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
                        _check_router_microstep(compiler, reference, x, y)
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        if actual.requires_grad:
                            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
                    ref_optimizer.step()
                    optimizer.step()
                    ref_monitor.on_step_end()
                    monitor.on_step_end()
                    for actual, expected in zip(model.state_dict().values(), reference.state_dict().values()):
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    self.assertEqual(int(model.layers[0].router.tokens_per_expert.sum()), 0)
                    ref_optimizer.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                if mode != "off":
                    self.assertTrue(any(node.meta.get("ac_recomputed_from")
                                        for node in compiler._joint_graph.graph_module.graph.nodes))

    def test_shared_producer_multiconsumer_replay(self) -> None:
        """Shared KV contributions accumulate into the producer across AC and new inputs."""
        for mode in ("off", "full"):
            with self.subTest(ac_mode=mode):
                torch.manual_seed(42)
                reference = _SharedModel()
                model = copy.deepcopy(reference)
                compiler = GraphCompiler(model, _train_fn, PassConfig(fsdp_enabled=False, ac_mode=mode),
                                         device=torch.device("cpu"))
                for _ in range(2):
                    x, y = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
                    expected_loss, _ = _train_fn(reference, x=x, y=y)
                    expected_loss.backward()
                    loss, _ = compiler.forward_backward(x=x, y=y)
                    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
