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
"""Real one-layer V4.1 dynamic/fixed eager/fixed graph acceptance."""

import importlib
import os
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.distributed.mesh import MeshContext
from tests.torch.expert_parallel.v41_graph_common import (
    average_gradients, compare_state, loss_fn, make_model,
)


_MHC_POST = importlib.import_module("hyper_parallel.components.functional.mhc_post")


def _step(models, optimizers, compiler, mesh, inputs, sharded):
    """Compare the entire loss/gradient/update chain for one actual token batch."""
    reference, fixed, graph = models
    with SkipDTensorDispatch():
        expected = loss_fn(reference, input_ids=inputs, labels=inputs)
        expected.backward()
        fixed_loss = loss_fn(fixed, input_ids=inputs, labels=inputs)
        fixed_loss.backward()
    actual, _ = compiler.forward_backward(input_ids=inputs, labels=inputs)
    assert torch.isfinite(actual).item()
    tolerance = {} if reference.dtype == torch.float32 else {"atol": 0.03, "rtol": 0.03}
    torch.testing.assert_close(fixed_loss, expected, **tolerance)
    torch.testing.assert_close(actual, expected, **tolerance)
    edp = mesh.fsdp_moe_mesh["edp_shard"]
    compare_state(reference, fixed, edp, gradients=True, sharded=False)
    if sharded:
        average_gradients(reference, edp)
        average_gradients(fixed, edp)
    gradients = compare_state(reference, graph, edp, gradients=True, sharded=sharded)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    weights = compare_state(reference, graph, edp, gradients=False, sharded=sharded)
    compare_state(reference, fixed, edp, gradients=False, sharded=False)
    return (expected.item(), fixed_loss.item(), actual.item()), gradients, weights


def _run_v41(sharded, dtype):
    """Use checked-in local fixtures; no model weights or tokenizer download."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch_npu.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        size = 4 if sharded else 2
        assert dist.get_world_size() == size
        mesh = MeshContext(dp_size=size, dp_shard_size=size, ep_size=2, edp_shard_size=size // 2)
        mesh.build_meshs("npu", size)
        torch.manual_seed(1024)
        models = [make_model(device, mesh.fsdp_moe_mesh["ep"], backend, dtype)
                  for backend in ("dynamic", "fixed_slot", "fixed_slot")]
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
            for left, right in zip(models[0].parameters(), model.parameters()):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        compiler = GraphCompiler(models[2], loss_fn,
                                 PassConfig(ep_enabled=True, fsdp_enabled=sharded, enable_overlap=False),
                                 mesh_context=mesh, device=device)
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
        _train(models, optimizers, compiler, mesh, device, sharded)
    finally:
        dist.destroy_process_group()


def _train(models, optimizers, compiler, mesh, device, sharded):
    """Observe real Router/shared calls throughout the declared precision contract."""
    events, handles = [], []
    steps = 10 if models[0].dtype == torch.float32 else 2
    for index, model in enumerate(models):
        mlp = model.model.layers[0].mlp
        for label, module in (("router", mlp.gate), ("shared", mlp.shared_experts)):
            handles.append(module.register_forward_pre_hook(lambda *_, key=(index, label): events.append(key)))
    try:
        # The optional Omni post kernel is not required for EP acceptance.
        with patch.object(_MHC_POST, "omni_training_custom_ops", None):
            for step in range(steps):
                inputs = torch.arange(1 + step, 17 + step, device=device).unsqueeze(0) + dist.get_rank()
                losses, gradients, weights = _step(models, optimizers, compiler, mesh, inputs, sharded)
                print(f"rank={dist.get_rank()} step={step} v41_losses={losses} "
                      f"grad_count={gradients} weight_count={weights}", flush=True)
        assert compiler.is_compiled
        for index in range(3):
            for label in ("router", "shared"):
                assert events.count((index, label)) == (1 if index == 2 else steps)
        print(f"rank={dist.get_rank()} v41_one_layer PASS fsdp={sharded} "
              f"dtype={models[0].dtype} steps={steps}", flush=True)
    finally:
        for handle in handles:
            handle.remove()


def test_v41_one_layer_ep():
    """Compare complete BF16 V4.1 pure EP for the existing two-update contract."""
    _run_v41(False, torch.bfloat16)


def test_v41_one_layer_fsdp_ep():
    """Compare complete BF16 V4.1 FSDP+EP for the existing two-update contract."""
    _run_v41(True, torch.bfloat16)


def test_v41_one_layer_ep_fp32():
    """Compare complete FP32 V4.1 pure EP for ten updates."""
    _run_v41(False, torch.float32)


def test_v41_one_layer_fsdp_ep_fp32():
    """Compare complete FP32 V4.1 FSDP+EP for ten updates."""
    _run_v41(True, torch.float32)
