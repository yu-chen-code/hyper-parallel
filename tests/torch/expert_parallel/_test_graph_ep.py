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

"""Explicit static EP workers; native dynamic EP supplies the reference."""

import importlib
import os
from datetime import timedelta
from typing import Callable, Sequence
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.compile.ep_collectives import ep_all_to_all as graph_ep_all_to_all
from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, ep_routed_forward
from hyper_parallel.compile.ep_compute import bind_static_expert_forward
from hyper_parallel.compile.expert_parallel import routed_forward_with_router


class _LocalExpertModel(nn.Module):
    """Route tokens to each rank's expert and restore source token order."""

    def __init__(
        self,
        exchange: Callable,
        send_counts: Sequence[int],
        recv_counts: Sequence[int],
    ) -> None:
        """Keep a rank-local expert and its fixed source/destination layout."""
        super().__init__()
        self.pre_proj = nn.Linear(8, 8, bias=False)
        self.local_expert = nn.Linear(8, 8, bias=False)
        self.exchange = exchange
        self.send_counts = send_counts
        self.recv_counts = recv_counts

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the local projection, dispatch, expert, and combine."""
        tokens = self.pre_proj(tokens)
        dispatched = self.exchange(tokens, self.send_counts, self.recv_counts, dist.group.WORLD)
        local_output = self.local_expert(dispatched).relu()
        restored = self.exchange(local_output, self.recv_counts, self.send_counts, dist.group.WORLD)
        return restored.tanh()



def _loss(model, *, tokens):
    return model(tokens).square().mean()



def _routed_loss(model, *, tokens):
    output = model(tokens)
    return output.float().square().mean(), {"routed_output": output}



def _assert_bf16_signal_close(actual, expected, name):
    """Scale absolute tolerance to the signal being compared."""
    reference = expected.detach().float()
    rms = reference.square().mean().sqrt().item()
    torch.testing.assert_close(
        actual, expected, atol=0.05 * rms, rtol=0.05,
        msg=lambda msg: f"{name}: {msg}",
    )



def _assert_zero_is_rejected(reference, name):
    """Ensure the BF16 comparison cannot accept a missing signal."""
    if reference.detach().float().norm().item() == 0:
        raise AssertionError(f"{name} reference is zero")
    try:
        _assert_bf16_signal_close(torch.zeros_like(reference), reference, name)
    except AssertionError:
        return
    raise AssertionError(f"{name} comparison accepted a zero result")



def _split_plan(rank, split_mode):
    if split_mode == "equal":
        return [2, 2], [2, 2]
    if split_mode == "uneven":
        return ([1, 3], [1, 2]) if rank == 0 else ([2, 2], [3, 2])
    if split_mode == "zero":
        return ([0, 4], [0, 2]) if rank == 0 else ([2, 2], [4, 2])
    raise ValueError(f"unknown split mode: {split_mode}")



def _device_and_backend(device_type):
    if device_type == "npu":
        # CPU-only test environments need not have the optional NPU package.
        importlib.import_module("torch_npu")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        return torch.device("npu", local_rank), "hccl"
    return torch.device("cpu"), "gloo"



def _compare_outputs(actual_pair, expected_pair, routed, bf16, check_zero):
    """Compare loss and routed outputs without accepting missing BF16 signals."""
    actual, actual_dict = actual_pair
    expected, expected_dict = expected_pair
    assert actual.dtype == expected.dtype
    if bf16:
        _assert_bf16_signal_close(actual, expected, "loss")
        _assert_bf16_signal_close(
            actual_dict["routed_output"], expected_dict["routed_output"], "output"
        )
        if check_zero:
            _assert_zero_is_rejected(expected_dict["routed_output"], "output")
    else:
        torch.testing.assert_close(actual, expected)
        if routed:
            torch.testing.assert_close(
                actual_dict["routed_output"], expected_dict["routed_output"]
            )



def _compare_gradients(eager, graph_model, routed, bf16, check_zero):
    """Compare every parameter gradient and require a nonzero routed signal."""
    nonzero_count = 0
    for (name, eager_param), graph_param in zip(eager.named_parameters(), graph_model.parameters()):
        nonzero_count += int(torch.count_nonzero(eager_param.grad).item())
        if bf16:
            _assert_bf16_signal_close(graph_param.grad, eager_param.grad, name)
            if check_zero and eager_param.grad.detach().float().norm().item() > 0:
                _assert_zero_is_rejected(eager_param.grad, name)
        else:
            torch.testing.assert_close(
                graph_param.grad, eager_param.grad, msg=lambda msg, name=name: f"{name}: {msg}"
            )
    if routed and nonzero_count == 0:
        raise AssertionError("Routed EP produced no nonzero gradients")



def _compare_updated_parameters(eager, graph_model, bf16):
    for (name, eager_param), graph_param in zip(eager.named_parameters(), graph_model.parameters()):
        if bf16:
            _assert_bf16_signal_close(graph_param, eager_param, name)
        else:
            torch.testing.assert_close(
                graph_param, eager_param, msg=lambda msg, name=name: f"{name}: {msg}"
            )



def _check_steps(eager, graph_model, compiler, device, rank, token_dtype=torch.float32,
                 routed=False):
    """Compare independent eager and graph updates over consecutive steps."""
    eager_optim = torch.optim.SGD(eager.parameters(), lr=0.1)
    graph_optim = torch.optim.SGD(graph_model.parameters(), lr=0.1)
    bf16 = token_dtype == torch.bfloat16 and routed
    for step in range(2):
        tokens = torch.randn(4, 8, device=device, dtype=token_dtype) + rank * 0.5
        expected_pair = (
            _routed_loss(eager, tokens=tokens)
            if routed else (_loss(eager, tokens=tokens), {})
        )
        expected_pair[0].backward()
        actual_pair = compiler.forward_backward(tokens=tokens)
        _compare_outputs(actual_pair, expected_pair, routed, bf16, step == 0)
        _compare_gradients(eager, graph_model, routed, bf16, step == 0)
        eager_optim.step()
        graph_optim.step()
        eager_optim.zero_grad()
        graph_optim.zero_grad()
        _compare_updated_parameters(eager, graph_model, bf16)



def _run_graph_ep(device_type, split_mode):
    """Run two-rank joint capture and replay with the selected fixed splits."""
    device, backend = _device_and_backend(device_type)

    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        send_counts, recv_counts = _split_plan(rank, split_mode)
        torch.manual_seed(73 + rank)
        eager = _LocalExpertModel(ep_all_to_all, send_counts, recv_counts).to(device)
        graph_model = _LocalExpertModel(graph_ep_all_to_all, send_counts, recv_counts).to(device)
        graph_model.load_state_dict(eager.state_dict())
        compiler = GraphCompiler(
            graph_model, _loss, PassConfig(fsdp_enabled=False, enable_overlap=False), device=device
        )
        _check_steps(eager, graph_model, compiler, device, rank)
        targets = [str(node.target) for node in compiler._joint_graph.graph_module.graph.nodes]
        assert sum("all_to_all_single" in target for target in targets) == 4
    finally:
        dist.destroy_process_group()



def test_graph_ep_gloo_equal():
    """Match eager EP with fixed equal splits on two CPU ranks."""
    _run_graph_ep("cpu", "equal")



def test_graph_ep_hccl_equal():
    """Match eager EP with fixed equal splits on two NPU ranks."""
    _run_graph_ep("npu", "equal")



def test_graph_ep_hccl_uneven():
    """Match eager EP with fixed uneven splits on two NPU ranks."""
    _run_graph_ep("npu", "uneven")



def test_graph_ep_hccl_zero():
    """Match eager EP when one peer receives no tokens on an NPU rank."""
    _run_graph_ep("npu", "zero")



def test_explicit_static_route_against_native_dynamic():
    """Compare static math to untouched dynamic dispatch over changing routes."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend, timeout=timedelta(seconds=90))
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(301)
            reference = nn.Module()
            reference.num_experts = 4
            reference.experts = _TinyRoutedExperts().to(device=device, dtype=dtype)
            bind_local_expert_forward(reference, ep_size=2)
            static = nn.Module()
            static.num_experts = 4
            static.experts = _TinyRoutedExperts().to(device=device, dtype=dtype)
            static.load_state_dict(reference.state_dict())
            bind_static_expert_forward(static, ep_size=2)
            optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in (reference, static)]
            for step in range(10):
                for accumulation in range(2):
                    hidden = torch.randn(1, 4, 8, device=device, dtype=dtype)
                    indices = torch.arange(8, device=device).reshape(4, 2) % (2 if (step + accumulation) % 2 else 4)
                    weights = torch.full((4, 2), 0.5, device=device, dtype=torch.float32)

                    def router(module: nn.Module, states: torch.Tensor,
                               route_indices: torch.Tensor = indices,
                               route_weights: torch.Tensor = weights) -> tuple:
                        """Return this call's fixed routing without capturing later iterations."""
                        del module, states
                        return route_indices, route_weights

                    expected = ep_routed_forward(reference, hidden, router_fn=router, ep_group=dist.group.WORLD)
                    actual = routed_forward_with_router(static, hidden, router_fn=router, ep_group=dist.group.WORLD)
                    if dtype == torch.bfloat16:
                        _assert_bf16_signal_close(actual, expected, "static output")
                    else:
                        torch.testing.assert_close(actual, expected)
                    expected.float().square().mean().backward()
                    actual.float().square().mean().backward()
                for left, right in zip(reference.parameters(), static.parameters()):
                    if dtype == torch.bfloat16:
                        _assert_bf16_signal_close(right.grad, left.grad, "static gradient")
                        _assert_zero_is_rejected(left.grad, "gradient")
                    else:
                        torch.testing.assert_close(right.grad, left.grad)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            print(f"rank={dist.get_rank()} static_route_reference PASS dtype={dtype}", flush=True)
    finally:
        dist.destroy_process_group()



class _TinyRoutedExperts(nn.Module):
    """Hold two local fused SwiGLU experts out of four global experts."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the same stacked weight layout used by eager EP."""
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 12, width) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(2, width, 6) * 0.1)
        self.act_fn = torch.nn.functional.silu
