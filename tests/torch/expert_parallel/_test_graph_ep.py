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
"""Two-rank joint-graph EP training against the existing eager collective."""

import importlib
import os
from datetime import timedelta
from typing import Callable, Sequence
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig, graph_ep_all_to_all
from hyper_parallel.compile.expert_parallel import routed_forward
from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_dispatch,
    ep_routed_experts_and_combine,
    ep_routed_forward,
)
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.compile.ep_dispatch import FixedSlotTokenDispatcher, reject_routing
from hyper_parallel.compile.ep_compute import bind_static_expert_forward
from hyper_parallel.compile.expert_parallel import routed_forward_with_router
from hyper_parallel import SkipDTensorDispatch


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


class _TinyRoutedExperts(nn.Module):
    """Hold two local fused SwiGLU experts out of four global experts."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the same stacked weight layout used by eager EP."""
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 12, width) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(2, width, 6) * 0.1)
        self.act_fn = torch.nn.functional.silu


class _TinyRoutedModel(nn.Module):
    """Use one forward definition for eager and graph routed EP."""

    def __init__(self, width: int = 8, dispatcher_backend: str = "fixed_slot") -> None:
        """Bind the eager expert contract before tracing either mode."""
        super().__init__()
        self.num_experts = 4
        self.router = nn.Linear(width, 4, bias=False)
        self.experts = _TinyRoutedExperts(width)
        self.ep_group = dist.group.WORLD
        self.static_ep = dispatcher_backend == "fixed_slot"
        if self.static_ep:
            bind_static_expert_forward(self, ep_size=2)
        else:
            bind_local_expert_forward(self, ep_size=2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Route two experts per token through the common EP entry."""
        states = tokens.unsqueeze(0)
        if self.static_ep:
            if getattr(self, "use_split_dispatch", False):
                reject_routing(
                    self.ep_group, "Graph EP does not support split dispatch with overlap_shared_expert=True"
                )
            return routed_forward_with_router(
                self, states, router_fn=_tiny_router, ep_group=self.ep_group
            ).squeeze(0)
        if getattr(self, "use_split_dispatch", False):
            state = ep_routed_dispatch(
                self, states, router_fn=_tiny_router, ep_group=self.ep_group
            )
            return ep_routed_experts_and_combine(self, state, self.ep_group).squeeze(0)
        return ep_routed_forward(
            self, states, router_fn=_tiny_router, ep_group=self.ep_group
        ).squeeze(0)


def _tiny_router(module, tokens):
    """Route Top-K tokens and inject the requested preflight failure."""
    if getattr(module, "force_router_error", False):
        raise RuntimeError("injected Router failure")
    router_input = tokens.reshape(-1, tokens.shape[-1]).to(module.router.weight.dtype)
    scores = module.router(router_input).softmax(dim=-1)
    weights, indices = scores.topk(2, dim=-1)
    if getattr(module, "route_from_tokens", False):
        skew = tokens.reshape(-1)[0] > 0
        forced = torch.arange(2, device=indices.device).expand_as(indices)
        indices = torch.where(skew, forced, indices)
    if getattr(module, "force_invalid_route", False):
        indices = indices + module.num_experts
    if getattr(module, "force_invalid_from_tokens", False):
        invalid = (tokens.reshape(-1)[0] > 0).to(dtype=indices.dtype)
        indices = indices + invalid * module.num_experts
    if getattr(module, "force_bad_routing_shape", False):
        weights = weights[:-1]
    return indices, weights


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


def test_graph_ep_hccl_routed():
    """Replay changed Top-K routes with one shared model forward on NPU."""
    _run_routed_ep(torch.float32)


def test_graph_ep_hccl_routed_bf16_fp32_router():
    """Preserve eager output dtype and gradients with BF16 experts and FP32 router."""
    _run_routed_ep(torch.bfloat16)
















def test_graph_ep_hccl_rejects_unequal_local_tokens():
    """Reject unequal rank-local capacities before entering routed all-to-all."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        torch.manual_seed(142)
        graph = _TinyRoutedModel().to(device)
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False),
            device=device,
        )
        tokens = torch.randn(4 - rank, 8, device=device)
        try:
            compiler.forward_backward(tokens=tokens)
        except ValueError as error:
            if "same local token count times Top-K" not in str(error):
                raise
        else:
            raise AssertionError("Unequal EP token capacities were accepted")
    finally:
        dist.destroy_process_group()


def test_graph_ep_hccl_rejects_hidden_width_mismatch():
    """Reject a peer's different hidden width during real HCCL graph capture."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        width = 8 if dist.get_rank() == 0 else 16
        graph = _TinyRoutedModel(width).to(device)
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False),
            device=device,
        )
        try:
            compiler.compile(tokens=torch.randn(4, width, device=device))
        except ValueError as error:
            assert "identical routing signatures" in str(error)
        else:
            raise AssertionError("HCCL capture accepted mismatched hidden widths")
        assert not compiler.is_compiled
    finally:
        dist.destroy_process_group()


def _run_rejected_split_overlap(rank_local: bool) -> None:
    """Agree on unsupported dispatch paths before either rank issues data exchange."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        graph = _TinyRoutedModel().to(device)
        graph.use_split_dispatch = not rank_local or dist.get_rank() == 0
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False),
            device=device,
        )
        with (
            patch("hyper_parallel.compile.ep_dispatch.ep_all_to_all") as graph_exchange,
            patch("hyper_parallel.distributed.expert_parallel.experts.ep_all_to_all_async") as eager_exchange,
        ):
            try:
                compiler.compile(tokens=torch.ones(4, 8, device=device))
            except ValueError as error:
                assert "rank 0" in str(error) and "overlap_shared_expert" in str(error)
            else:
                raise AssertionError("Graph capture accepted eager split dispatch")
            graph_exchange.assert_not_called()
            eager_exchange.assert_not_called()
        assert not compiler.is_compiled
    finally:
        dist.destroy_process_group()


def test_graph_ep_hccl_rejects_split_overlap():
    """Reject split dispatch consistently when both ranks enable overlap."""
    _run_rejected_split_overlap(rank_local=False)


def test_graph_ep_hccl_agrees_on_split_dispatch_mismatch():
    """Reject rank 0's split route while rank 1 uses the normal graph route."""
    _run_rejected_split_overlap(rank_local=True)


def _check_preparation_failure(case: str, device: torch.device) -> None:
    """Inject a rank-0 Router/weight error before the unified preparation gate."""
    graph = _TinyRoutedModel().to(device)
    if dist.get_rank() == 0:
        if case == "router":
            graph.force_router_error = True
        elif case == "missing_down":
            del graph.experts.down_proj
        else:
            shape = (1, 8, 6) if case == "local_count" else (2, 7, 6)
            graph.experts.down_proj = nn.Parameter(torch.ones(shape, device=device))
    compiler = GraphCompiler(
        graph, _routed_loss,
        PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False), device=device,
    )
    with patch("hyper_parallel.compile.ep_dispatch.ep_all_to_all") as exchange:
        try:
            compiler.compile(tokens=torch.ones(4, 8, device=device))
        except ValueError as error:
            assert "rank 0" in str(error)
            reason = {"router": "injected Router failure", "missing_down": "SwiGLU",
                      "local_count": "local_expert_count", "hidden_width": "input/output width"}[case]
            assert reason in str(error), str(error)
        else:
            raise AssertionError(f"Graph capture accepted rank-local failure: {case}")
        exchange.assert_not_called()
    assert not compiler.is_compiled


def test_graph_ep_hccl_agrees_on_preparation_failures():
    """Share rank-local Router and weight failures before the first data exchange."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        for case in ("router", "missing_down", "local_count", "hidden_width"):
            _check_preparation_failure(case, device)
    finally:
        dist.destroy_process_group()


def _check_tensor_contract_failure(case, device, capture):
    """Inject one rank's operand mismatch and require zero data exchanges."""
    model = _TinyRoutedModel().to(device)
    rank = dist.get_rank()
    if rank == 0 and case.startswith("expert"):
        if case == "expert_dtype":
            model.experts.to(dtype=torch.bfloat16)
        else:
            model.experts.cpu()
    original = _tiny_router

    def router(module: nn.Module, hidden: torch.Tensor) -> tuple:
        """Keep the normal Router except for a single rank's malformed operand."""
        indices, weights = original(module, hidden)
        if rank == 0 and case == "indices_device":
            indices = indices.cpu()
        if rank == 0 and case == "weights_device":
            weights = weights.cpu()
        if rank == 0 and case == "weights_dtype":
            weights = weights.to(torch.int64)
        return indices, weights

    compiler = GraphCompiler(model, _routed_loss,
                             PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False), device=device)
    reason = "dtype" if case == "expert_dtype" else "floating-point" if case == "weights_dtype" else "device"
    with (
        patch(__name__ + "._tiny_router", side_effect=router),
        patch("hyper_parallel.compile.ep_dispatch.ep_all_to_all") as exchange,
    ):
        try:
            tokens = torch.ones(4, 8, device=device)
            if capture:
                compiler.compile(tokens=tokens)
            else:
                _routed_loss(model, tokens=tokens)
        except ValueError as error:
            assert "rank 0" in str(error) and reason in str(error), str(error)
        else:
            raise AssertionError(f"Accepted rank-local operand mismatch: {case}")
        exchange.assert_not_called()


def test_graph_ep_hccl_agrees_on_tensor_contract_failures():
    """Strict eager and capture agree on unilateral dtype/device failures."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        for capture in (False, True):
            for case in ("expert_dtype", "expert_device", "indices_device", "weights_device", "weights_dtype"):
                _check_tensor_contract_failure(case, device, capture)
        print(f"rank={dist.get_rank()} tensor_contract_failures PASS", flush=True)
    finally:
        dist.destroy_process_group()


def test_graph_ep_hccl_agrees_on_local_capture_failure():
    """Share one rank's routing error during HCCL joint-graph capture."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        graph = _TinyRoutedModel().to(device)
        graph.force_bad_routing_shape = dist.get_rank() == 0
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False),
            device=device,
        )
        try:
            compiler.compile(tokens=torch.randn(4, 8, device=device))
        except ValueError as error:
            assert "rank 0" in str(error) and "matching Top-K" in str(error)
        else:
            raise AssertionError("HCCL capture accepted a rank-local routing error")
        assert not compiler.is_compiled
    finally:
        dist.destroy_process_group()


def test_graph_ep_gloo_agrees_on_local_validation_failure():
    """All ranks receive one rank's budget error before routed all-to-all."""
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        experts = nn.Module()
        experts.num_experts = 4
        experts.local_expert_count = 2
        if rank == 0:
            hidden = torch.empty(1, 1025, 4096, device="meta", dtype=torch.bfloat16)
            indices = torch.empty(1025, 8, device="meta", dtype=torch.long)
            weights = torch.empty(1025, 8, device="meta")
        else:
            hidden = torch.ones(1, 3, 8)
            indices = torch.zeros(3, 2, dtype=torch.long)
            weights = torch.ones(3, 2)
        try:
            routed_forward(experts, hidden, indices, weights, dist.group.WORLD)
        except ValueError as error:
            assert "rank 0" in str(error) and "dense dispatch requires" in str(error)
        else:
            raise AssertionError("Local routing failure did not reach every rank")
    finally:
        dist.destroy_process_group()


def test_graph_fsdp_ep_hccl_rejects_unsupported_meshes():
    """Reject expert replicas, dense HSDP, and TP before graph tracing."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        cases = (
            ({"dp_size": 4, "dp_shard_size": 4, "ep_size": 2, "edp_shard_size": 1},
             "edp_replicate"),
            ({"dp_size": 4, "dp_replicate_size": 2, "dp_shard_size": 2,
              "ep_size": 2, "edp_shard_size": 2}, "dense HSDP"),
            ({"dp_size": 2, "dp_shard_size": 2, "tp_size": 2,
              "ep_size": 2, "edp_shard_size": 2}, "tp_size"),
        )
        for mesh_config, reason in cases:
            mesh = MeshContext(**mesh_config)
            mesh.build_meshs("npu", 4)
            model = nn.Linear(8, 8, bias=False).to(device)
            compiler = GraphCompiler(
                model, _loss,
                PassConfig(fsdp_enabled=True, ep_enabled=True, enable_overlap=False),
                mesh_context=mesh, device=device,
            )
            try:
                compiler.compile(tokens=torch.ones(2, 8, device=device))
            except ValueError as error:
                assert reason in str(error), str(error)
            else:
                raise AssertionError(f"Unsupported mesh accepted: {mesh_config}")
            assert not compiler.is_compiled
    finally:
        dist.destroy_process_group()


def test_graph_fsdp_ep_hccl_rejects_nondivisible_experts():
    """Reject two local experts on an EDP shard axis of size four."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        mesh = MeshContext(dp_size=8, dp_shard_size=8, ep_size=2, edp_shard_size=4)
        mesh.build_meshs("npu", 8)
        graph = _TinyRoutedModel().to(device)
        graph.ep_group = mesh.fsdp_moe_mesh["ep"].get_group()
        original_shape = tuple(graph.experts.gate_up_proj.shape)
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=True, ep_enabled=True, enable_overlap=False),
            mesh_context=mesh, device=device,
        )
        try:
            compiler.compile(tokens=torch.ones(4, 8, device=device))
        except ValueError as error:
            assert "edp_shard degree 4" in str(error)
        else:
            raise AssertionError("Unsynchronized expert replicas were accepted")
        assert not compiler.is_compiled
        assert tuple(graph.experts.gate_up_proj.shape) == original_shape
    finally:
        dist.destroy_process_group()


def test_graph_fsdp_ep_hccl_routed():
    """Compare a four-rank joint FSDP+EP graph with reduced eager gradients."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        mesh = MeshContext(dp_size=4, dp_shard_size=4, ep_size=2, edp_shard_size=2)
        mesh.build_meshs("npu", 4)
        ep_group = mesh.fsdp_moe_mesh["ep"].get_group()
        expert_group = mesh.fsdp_moe_mesh["edp_shard"].get_group()
        torch.manual_seed(314)
        eager = _TinyRoutedModel(dispatcher_backend="dynamic").to(device)
        graph = _TinyRoutedModel().to(device)
        graph.load_state_dict(eager.state_dict())
        eager.ep_group = ep_group
        graph.ep_group = ep_group
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=True, ep_enabled=True, enable_overlap=False),
            mesh_context=mesh, device=device,
        )
        eager_optim = torch.optim.SGD(eager.parameters(), lr=0.1)
        graph_optim = torch.optim.SGD(graph.parameters(), lr=0.1)
        for _ in range(2):
            tokens = torch.randn(4, 8, device=device) + rank * 0.2
            expected, expected_dict = _routed_loss(eager, tokens=tokens)
            expected.backward()
            actual, actual_dict = compiler.forward_backward(tokens=tokens)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                actual_dict["routed_output"], expected_dict["routed_output"]
            )
            for name, eager_param in eager.named_parameters():
                graph_param = dict(graph.named_parameters())[name]
                is_expert = name.startswith("experts.")
                group = expert_group if is_expert else dist.group.WORLD
                degree = 2 if is_expert else 4
                dist.all_reduce(eager_param.grad, group=group)
                eager_param.grad.div_(degree)
                shard = eager_param.grad.chunk(degree, dim=0)[dist.get_rank(group)]
                torch.testing.assert_close(graph_param.grad, shard, msg=name)
            eager_optim.step()
            graph_optim.step()
            eager_optim.zero_grad()
            graph_optim.zero_grad()
            for name, eager_param in eager.named_parameters():
                graph_param = dict(graph.named_parameters())[name]
                is_expert = name.startswith("experts.")
                group = expert_group if is_expert else dist.group.WORLD
                degree = 2 if is_expert else 4
                shard = eager_param.chunk(degree, dim=0)[dist.get_rank(group)]
                torch.testing.assert_close(graph_param, shard, msg=name)
        assert compiler.is_compiled
        groups = {
            node.meta.get("comm_group")
            for node in compiler._joint_graph.graph_module.graph.nodes
        }
        assert "fsdp" in groups and "fsdp_expert" in groups
    finally:
        dist.destroy_process_group()


def test_graph_ep_hccl_invalid_replay_is_nonfinite():
    """One rank's invalid replay route must stop every rank before the optimizer."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        graph = _TinyRoutedModel().to(device)
        graph.force_invalid_from_tokens = True
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False),
            device=device,
        )
        compiler.compile(tokens=torch.full((4, 8), -1.0, device=device))
        replay_tokens = torch.full((4, 8), 1.0 if rank == 0 else -1.0, device=device)
        local_loss, _, values = compiler._run_graph(tokens=replay_tokens)
        assert torch.isnan(local_loss).item() == (rank == 0)
        assert torch.isnan(values["routed_output"]).all().item() == (rank == 0)
        try:
            compiler.forward_backward(tokens=replay_tokens)
        except FloatingPointError as error:
            assert "at least one rank" in str(error)
        else:
            raise AssertionError("One invalid EP rank did not stop every rank")
        targets = [str(node.target) for node in compiler._joint_graph.graph_module.graph.nodes]
        assert not any("_assert_async" in target for target in targets)
    finally:
        dist.destroy_process_group()


def test_graph_ep_hccl_independent_subgroup_steps():
    """An idle EP subgroup must not block another group's loss/error check."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend, timeout=timedelta(seconds=90))
    try:
        rank = dist.get_rank()
        first = dist.new_group([0, 1])
        second = dist.new_group([2, 3])
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
        graph = _TinyRoutedModel().to(device)
        graph.ep_group = first if rank < 2 else second
        graph.force_invalid_from_tokens = rank < 2
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False), device=device,
        )
        signal = torch.zeros(1)
        if rank < 2:
            compiler.compile(tokens=torch.full((4, 8), -1.0, device=device))
            try:
                compiler.forward_backward(
                    tokens=torch.full((4, 8), 1.0 if rank == 0 else -1.0, device=device)
                )
            except FloatingPointError:
                assert all(param.grad is None for param in graph.parameters())
            else:
                raise AssertionError("Invalid replay did not stop both EP-A ranks")
        # EP-B waits here while EP-A captures and checks its invalid replay.
        dist.broadcast(signal, src=0, group=control)
        if rank >= 2:
            for _ in range(2):
                loss, _ = compiler.forward_backward(tokens=torch.randn(4, 8, device=device))
                assert torch.isfinite(loss).item()
                graph.zero_grad(set_to_none=True)
        dist.broadcast(signal, src=2, group=control)
        assert compiler._ep_sync_group is graph.ep_group
    finally:
        dist.destroy_process_group()


def _run_routed_ep(hidden_dtype):
    """Compare the common routed forward under the requested activation dtype."""
    device, backend = _device_and_backend("npu")
    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        torch.manual_seed(142)
        eager = _TinyRoutedModel(dispatcher_backend="dynamic").to(device)
        graph = _TinyRoutedModel().to(device)
        eager.experts.to(dtype=hidden_dtype)
        graph.experts.to(dtype=hidden_dtype)
        graph.load_state_dict(eager.state_dict())
        _, router_weights = _tiny_router(
            eager, torch.zeros(1, 4, 8, device=device, dtype=hidden_dtype)
        )
        assert router_weights.dtype == torch.float32
        compiler = GraphCompiler(
            graph, _routed_loss,
            PassConfig(fsdp_enabled=False, ep_enabled=True, enable_overlap=False),
            device=device,
        )
        _check_steps(eager, graph, compiler, device, rank,
                     token_dtype=hidden_dtype, routed=True)
        assert compiler.is_compiled
        targets = [str(node.target) for node in compiler._joint_graph.graph_module.graph.nodes]
        assert any("where" in target for target in targets)
    finally:
        dist.destroy_process_group()


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
