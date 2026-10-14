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
"""Real one-layer V4.1 model and three-way state comparisons for EP ST."""

from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist

from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_ep_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (
    build_deepseek_v41_validation_config,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM


def make_model(device: torch.device, ep_mesh: object, backend: str, dtype: torch.dtype) -> torch.nn.Module:
    """Create a real V4.1 decoder with learned routing, clamp and shared experts.

    Args:
        device: Local NPU device.
        ep_mesh: Expert-axis mesh of size two.
        backend: Explicit dynamic or fixed_slot dispatcher selection.
        dtype: FP32 for ten updates or BF16 for the existing two-update contract.

    Returns:
        Causal language model with one complete validation decoder.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures/v41_one_layer"
    config = build_deepseek_v41_validation_config(
        str(fixtures), str(fixtures / "engram.json"), text_parameter_divisor=8,
        num_routed_experts=8, exercise_post_training_indexer=False,
    )
    assert (config.num_hidden_layers, config.hidden_size, config.num_local_experts) == (1, 640, 8)
    model = DeepseekV41ForCausalLM(config).to(device=device, dtype=dtype)
    for layer in model.model.layers:
        mlp = layer.mlp
        for parameter in mlp.experts.parameters():
            count = parameter.shape[0] // ep_mesh.size()
            parameter.data = parameter.detach().narrow(0, ep_mesh.get_local_rank() * count, count).clone()
        compute = deepseek_v41_ep_compute_fn(
            module=mlp, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=ep_mesh,
            dispatcher_backend=backend, use_grouped_gemm=False,
        )
        mlp.forward = MethodType(compute, mlp)
        assert mlp.experts.limit > 0 and mlp.experts._ep_apply_gate == mlp.experts._apply_gate
        assert not mlp.is_hash and mlp.shared_experts is not None
    return model


def loss_fn(model: torch.nn.Module, *, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the real causal language-model training loss.

    Args:
        model: Complete one-layer V4.1 model.
        input_ids: Integer token IDs, shape [batch, sequence].
        labels: Matching next-token labels.

    Returns:
        Scalar language-model loss.
    """
    return model(input_ids=input_ids, labels=labels).loss


def compare_state(reference: torch.nn.Module, candidate: torch.nn.Module,
                  expert_mesh: object, gradients: bool, sharded: bool) -> int:
    """Compare local or sharded gradients/weights using the existing BF16 contract.

    Args:
        reference: Dynamic eager model, already averaged when sharded is true.
        candidate: Fixed eager model or the graph FSDP model.
        expert_mesh: EDP shard axis used by expert parameters.
        gradients: Compare gradients instead of updated parameters.
        sharded: Select the corresponding reference shard for the graph model.

    Returns:
        Number of nonempty parameter comparisons.
    """
    count = 0
    tolerance = {} if reference.dtype == torch.float32 else {"atol": 0.03, "rtol": 0.03}
    for (name, left), (other_name, right) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert name == other_name
        expected, actual = (left.grad, right.grad) if gradients else (left, right)
        if expected is None or actual is None:
            present = actual if expected is None else expected
            if present is not None:
                torch.testing.assert_close(present, torch.zeros_like(present), atol=0, rtol=0, msg=name)
            continue
        group = expert_mesh.get_group() if ".experts." in name else dist.group.WORLD
        degree = dist.get_world_size(group) if sharded else 1
        if expected.ndim and expected.shape[0] % degree == 0 and degree > 1:
            expected = expected.chunk(degree, dim=0)[dist.get_rank(group)]
        torch.testing.assert_close(actual, expected, **tolerance, msg=name)
        count += 1
    assert count >= 25, count
    return count


def average_gradients(model: torch.nn.Module, expert_mesh: object) -> None:
    """Average full eager gradients to match the graph's supported FSDP plan.

    Args:
        model: Independent eager reference with complete dense/EP-local weights.
        expert_mesh: EDP shard group; dense parameters use the world FSDP group.
    """
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = expert_mesh.get_group() if ".experts." in name else dist.group.WORLD
        degree = dist.get_world_size(group)
        if parameter.ndim and parameter.shape[0] % degree == 0 and degree > 1:
            dist.all_reduce(parameter.grad, group=group)
            parameter.grad.div_(degree)
