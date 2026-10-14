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
"""Reject invalid graph EP plans before entering a distributed collective."""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.expert_parallel import (
    GraphEPCaptureContext,
    ep_all_to_all,
    graph_ep_trace,
    routed_forward,
)
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.distributed.expert_parallel.dispatch_contract import ExpertBatch
from hyper_parallel.distributed.expert_parallel.dispatchers import FixedSlotTokenDispatcher
from hyper_parallel.distributed.expert_parallel.experts import (
    _GRAPH_EP_CAPTURE,
    _local_swiglu_expert_forward,
    compute_expert_major,
    ep_routed_dispatch,
    resolve_swiglu_weights,
    execute_routed_experts,
    swiglu_expert_forward,
)
from hyper_parallel.distributed.mesh import MeshContext


_DISPATCHER_MODULE = "hyper_parallel.distributed.expert_parallel.dispatchers"


def _tanh_gate(projected):
    """Apply the custom fused gate used by the independent reference."""
    return projected[..., :3].tanh() * projected[..., 3:]


def _make_routed_experts():
    """Build a valid two-expert shard for routing validation tests."""
    experts = torch.nn.Module()
    experts.local_expert_count = 2
    experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 12, 8))
    experts.down_proj = torch.nn.Parameter(torch.randn(2, 8, 6))
    return experts


def _make_swiglu_experts(layout, dtype, custom_gate):
    """Build stacked weights without relying on the shared compute helper."""
    experts = torch.nn.Module()
    if layout == "fused":
        experts.gate_up_proj = torch.nn.Parameter(torch.randn(1, 6, 4, dtype=dtype) * 0.4)
        experts.down_proj = torch.nn.Parameter(torch.randn(1, 4, 3, dtype=dtype) * 0.4)
        if custom_gate:
            experts._ep_apply_gate = _tanh_gate
    else:
        experts.w1 = torch.nn.Parameter(torch.randn(1, 3, 4, dtype=dtype) * 0.4)
        experts.w3 = torch.nn.Parameter(torch.randn(1, 3, 4, dtype=dtype) * 0.4)
        experts.w2 = torch.nn.Parameter(torch.randn(1, 4, 3, dtype=dtype) * 0.4)
        experts._ep_act_fn = F.gelu
    return experts


def _reference_swiglu(experts, states, layout, custom_gate):
    """Compute the independent mathematical reference for one expert."""
    if layout == "fused":
        projected = F.linear(states, experts.gate_up_proj[0])
        gate, up = projected.chunk(2, dim=-1)
        activated = (
            gate.tanh() * up if custom_gate else F.silu(gate) * up
        )
        return F.linear(activated, experts.down_proj[0])
    gate = F.linear(states, experts.w1[0])
    up = F.linear(states, experts.w3[0])
    activated = F.gelu(gate) * up
    return F.linear(activated, experts.w2[0])


class TestGraphExpertParallelValidation(unittest.TestCase):
    """Local shape mistakes must fail before peers can wait in all-to-all."""

    @staticmethod
    def _gather_local_plan(output, plan, group):
        """Mirror the local status on both mocked ranks."""
        del group
        output[:] = [plan, plan]

    def test_dynamic_ep_rejected_before_tracing(self):
        """Preserve the static EP contract when upstream dynamic tracing is enabled."""
        for options in ({"dynamic": True}, {"dynamic_arg_dims": {"tokens": [0]}}):
            with self.subTest(options=options):
                compiler = GraphCompiler(
                    torch.nn.Linear(8, 8), lambda *_: None,
                    PassConfig(fsdp_enabled=False, ep_enabled=True),
                    device=torch.device("cpu"), **options,
                )
                with patch("hyper_parallel.compile.compiler.trace_model_graph") as trace:
                    with self.assertRaisesRegex(ValueError, "requires static shapes"):
                        compiler.compile(tokens=torch.ones(2, 8))
                    trace.assert_not_called()

    def test_invalid_plan_never_enters_collective(self):
        """Reject count and token-row mismatches without issuing communication."""
        group = object()
        tokens = torch.randn(4, 8)
        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                  side_effect=self._gather_local_plan),
            patch(_DISPATCHER_MODULE + ".fc.all_to_all_single_autograd") as collective,
        ):
            with self.assertRaisesRegex(ValueError, "send_counts"):
                ep_all_to_all(tokens, [4], [2, 2], group)
            with self.assertRaisesRegex(ValueError, "EP token rows"):
                ep_all_to_all(tokens, [1, 2], [2, 2], group)
            collective.assert_not_called()

    def test_graph_route_is_scoped_to_capture(self):
        """Restore eager routing even when graph capture raises an error."""
        self.assertIsNone(_GRAPH_EP_CAPTURE.get())
        with graph_ep_trace(True) as outer:
            self.assertIs(_GRAPH_EP_CAPTURE.get(), outer)
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with graph_ep_trace(True) as inner:
                    self.assertIs(_GRAPH_EP_CAPTURE.get(), inner)
                    self.assertIsNot(inner, outer)
                    raise RuntimeError("capture failed")
            self.assertIs(_GRAPH_EP_CAPTURE.get(), outer)
        self.assertIsNone(_GRAPH_EP_CAPTURE.get())

    def test_graph_rejects_split_overlap_dispatch(self):
        """A captured split route must fail before eager async exchange."""
        with (
            graph_ep_trace(True),
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                  side_effect=self._gather_local_plan) as metadata,
            patch("hyper_parallel.distributed.expert_parallel.experts.ep_all_to_all_async") as exchange,
        ):
            with self.assertRaisesRegex(ValueError, "rank 0.*overlap_shared_expert"):
                ep_routed_dispatch(None, torch.randn(2, 8), router_fn=lambda *_: None,
                                   ep_group=object())
            metadata.assert_called_once()
            exchange.assert_not_called()

    def test_graph_route_receives_peer_split_dispatch_failure(self):
        """A normal-route rank rejects its split-dispatch peer before data exchange."""
        experts = _make_routed_experts()
        hidden = torch.randn(2, 8)
        indices = torch.zeros(2, 1, dtype=torch.long)
        weights = torch.ones(2, 1)

        def gather_peer_error(output: list, plan: tuple, group: object) -> None:
            """Pair the valid local plan with the unsupported peer path."""
            del group
            output[:] = [(None, "Graph EP does not support split dispatch with overlap_shared_expert=True"), plan]

        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object", side_effect=gather_peer_error),
            patch(_DISPATCHER_MODULE + ".fc.all_to_all_single_autograd") as exchange,
        ):
            with self.assertRaisesRegex(ValueError, "rank 0.*overlap_shared_expert"):
                routed_forward(experts, hidden, indices, weights, object())
            exchange.assert_not_called()

    def test_graph_ep_rejects_mixed_parallelism(self):
        """Allow FSDP+EP while excluding unsupported axes and DP modes."""
        self.assertTrue(PassConfig(ep_enabled=True, fsdp_enabled=True).fsdp_enabled)
        with self.assertRaisesRegex(ValueError, "FSDP\\+EP"):
            PassConfig(ep_enabled=True, fsdp_enabled=True, dp_mode="hsdp")
        with self.assertRaisesRegex(ValueError, "FSDP\\+EP"):
            PassConfig(ep_enabled=True, fsdp_enabled=False, tp_size=2)

    def test_router_exception_joins_preflight(self):
        """A user Router exception is shared before the graph issues data exchange."""
        module = torch.nn.Module()
        module.experts = _make_routed_experts()
        router = unittest.mock.Mock(side_effect=KeyError("Router failed"))
        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                  side_effect=self._gather_local_plan) as metadata,
            patch(_DISPATCHER_MODULE + ".ep_all_to_all") as exchange,
        ):
            with self.assertRaisesRegex(ValueError, "rank 0.*KeyError.*Router failed"):
                group = object()
                execute_routed_experts(module.experts, torch.ones(2, 8), router,
                                       FixedSlotTokenDispatcher(group, 2), group)
            metadata.assert_called_once()
            exchange.assert_not_called()

    def test_bad_expert_weights_fail_before_data_exchange(self):
        """Missing weights, wrong shard counts and incompatible dimensions agree first."""
        cases = (None, (1, 8, 6), (2, 7, 6), (2, 8, 5), (2, 8))
        for shape in cases:
            with self.subTest(shape=shape):
                experts = _make_routed_experts()
                if shape is None:
                    del experts.down_proj
                else:
                    experts.down_proj = torch.nn.Parameter(torch.empty(shape))
                with (
                    patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
                    patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
                    patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                          side_effect=self._gather_local_plan),
                    patch(_DISPATCHER_MODULE + ".ep_all_to_all") as exchange,
                ):
                    with self.assertRaisesRegex(ValueError, "rank 0"):
                        routed_forward(experts, torch.ones(2, 8), torch.zeros(2, 1, dtype=torch.long),
                                       torch.ones(2, 1), object())
                    exchange.assert_not_called()

    def test_expert_classification_excludes_auxiliary_submodule(self):
        """Only resolved SwiGLU weights receive the expert shard group."""
        model = torch.nn.Module()
        model.experts = _make_routed_experts()
        model.experts.auxiliary = torch.nn.Linear(8, 8)
        context = GraphEPCaptureContext(model)
        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
        ):
            context.record_weights(resolve_swiglu_weights(model.experts))
        expected = {"experts.gate_up_proj", "experts.down_proj"}
        self.assertEqual(context.expert_param_names, expected)
        compiler = GraphCompiler(model, lambda *_: None, PassConfig(fsdp_enabled=True, ep_enabled=True))
        compiler._ep_expert_param_names = context.expert_param_names
        self.assertEqual(compiler._build_pass_kwargs()["expert_param_names"], expected)

    def test_low_level_exchange_does_not_register_capture_groups(self):
        """Group tracking belongs to successful routing preparation."""
        with (
            graph_ep_trace(True) as context,
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".fc.all_to_all_single_autograd"),
        ):
            ep_all_to_all(torch.ones(2, 8), [1, 1], [1, 1], object())
            self.assertEqual(context.groups, [])

    def test_eager_counts_use_host_prefix_sum(self):
        """Avoid device cumsum and preserve row order, outputs and gradients."""
        experts = _make_routed_experts()
        reference = copy.deepcopy(experts)
        tokens = torch.randn(3, 8, requires_grad=True)
        reference_tokens = tokens.detach().clone().requires_grad_()
        indices = torch.tensor([1, 0, 0])
        with patch.object(torch.Tensor, "cumsum", side_effect=AssertionError("device cumsum")):
            actual = _local_swiglu_expert_forward(experts, tokens, indices)
        outputs = []
        for row, index in enumerate((1, 0, 0)):
            projected = reference_tokens[row:row + 1] @ reference.gate_up_proj[index].t()
            gate, up = projected.chunk(2, dim=-1)
            outputs.append((gate * gate.sigmoid() * up) @ reference.down_proj[index].t())
        expected = torch.cat(outputs)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(tokens.grad, reference_tokens.grad)
        for parameter, reference_parameter in zip(experts.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)

    def test_graph_ep_rejects_unsupported_mesh_axes(self):
        """Check actual mesh topology even when PassConfig uses default axes."""
        axes = (
            ("tp_size", 2),
            ("pp_size", 2),
            ("cp_size", 2),
            ("sequence_parallel", True),
            ("loss_parallel", True),
        )
        for axis, value in axes:
            for fsdp in (False, True):
                with self.subTest(axis=axis, fsdp=fsdp):
                    mesh = MeshContext(**{axis: value})
                    compiler = GraphCompiler(
                        torch.nn.Linear(8, 8), lambda model, **inputs: model(**inputs),
                        PassConfig(fsdp_enabled=fsdp, ep_enabled=True),
                        mesh_context=mesh, device=torch.device("cpu"),
                    )
                    with patch("hyper_parallel.compile.compiler.trace_model_graph") as trace:
                        with self.assertRaisesRegex(ValueError, axis):
                            compiler.compile(tokens=torch.ones(2, 8))
                        trace.assert_not_called()

    def test_loss_sync_group_covers_captured_ep_participants(self):
        """Pure EP uses its subgroup; FSDP+EP covers it with the dense group."""
        ep, alias, other, dense = object(), object(), object(), object()
        ranks = {ep: [0, 1], alias: [0, 1], other: [2, 3], dense: [0, 1, 2, 3]}
        compiler = GraphCompiler(
            torch.nn.Linear(8, 8), lambda *_: None,
            PassConfig(fsdp_enabled=False, ep_enabled=True), device=torch.device("cpu"),
        )
        with patch("hyper_parallel.compile.compiler.dist.get_process_group_ranks", side_effect=ranks.__getitem__):
            compiler._resolve_ep_sync_group([ep, alias])
            self.assertIs(compiler._ep_sync_group, ep)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                compiler._resolve_ep_sync_group([ep, other])
            compiler.pass_config.fsdp_enabled = True
            compiler._ep_sync_group = dense
            compiler._resolve_ep_sync_group([ep])
            self.assertIs(compiler._ep_sync_group, dense)
            compiler._ep_sync_group = ep
            with self.assertRaisesRegex(ValueError, "incompatible"):
                compiler._resolve_ep_sync_group([other])

    def test_finite_loss_reduction_uses_resolved_subgroup(self):
        """The step must not introduce communication with unrelated ranks."""
        group = object()
        compiler = GraphCompiler(
            torch.nn.Linear(8, 8), lambda *_: None,
            PassConfig(fsdp_enabled=False, ep_enabled=True), device=torch.device("cpu"),
        )
        compiler._joint_graph = object()
        compiler._ep_sync_group = group
        with (
            patch.object(compiler, "_run_graph", return_value=(torch.ones(()), [], {})),
            patch.object(compiler, "_accumulate_grads"),
            patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=True),
            patch("hyper_parallel.compile.compiler.dist.all_reduce") as reduce,
        ):
            compiler.forward_backward()
            self.assertIs(reduce.call_args.kwargs["group"], group)

    def test_shared_swiglu_matches_independent_forward_and_backward(self):
        """Compare each input and weight gradient against an independent formula."""
        for layout, dtype, custom_gate in (
            ("separate", torch.float32, False),
            ("separate", torch.bfloat16, False),
            ("fused", torch.float32, False),
            ("fused", torch.bfloat16, False),
            ("fused", torch.float32, True),
            ("fused", torch.bfloat16, True),
        ):
            with self.subTest(layout=layout, dtype=dtype, custom_gate=custom_gate):
                torch.manual_seed(29)
                experts = _make_swiglu_experts(layout, dtype, custom_gate)
                reference = copy.deepcopy(experts)
                states = torch.randn(5, 4, dtype=dtype, requires_grad=True)
                reference_states = states.detach().clone().requires_grad_()
                actual = swiglu_expert_forward(
                    experts, states, 0, resolve_swiglu_weights(experts)
                )
                expected = _reference_swiglu(reference, reference_states, layout, custom_gate)
                upstream = torch.randn_like(actual)
                (actual * upstream).sum().backward()
                (expected * upstream).sum().backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(states.grad, reference_states.grad, rtol=0, atol=0)
                for (name, parameter), (reference_name, reference_parameter) in zip(
                    experts.named_parameters(), reference.named_parameters()
                ):
                    self.assertEqual(name, reference_name)
                    torch.testing.assert_close(
                        parameter.grad, reference_parameter.grad, rtol=0, atol=0
                    )
                    self.assertGreater(torch.count_nonzero(reference_parameter.grad).item(), 0)
                self.assertGreater(torch.count_nonzero(reference_states.grad).item(), 0)
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(torch.zeros_like(actual), expected, rtol=0, atol=0)
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(
                        torch.zeros_like(states.grad), reference_states.grad, rtol=0, atol=0
                    )

    def test_shared_swiglu_accepts_zero_token_expert(self):
        """An empty expert group keeps its shape and produces zero gradients."""
        experts = torch.nn.Module()
        experts.gate_up_proj = torch.nn.Parameter(torch.randn(1, 6, 4))
        experts.down_proj = torch.nn.Parameter(torch.randn(1, 4, 3))
        states = torch.empty(0, 4, requires_grad=True)
        output = swiglu_expert_forward(experts, states, 0, resolve_swiglu_weights(experts))
        self.assertEqual(tuple(output.shape), (0, 4))
        output.sum().backward()
        self.assertEqual(tuple(states.grad.shape), (0, 4))
        for parameter in experts.parameters():
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_expert_major_batch_dynamic_and_fixed_capacity(self):
        """Use one expert-major compute entry for ragged and padded rows."""
        for sizes, static in (((2, 1), False), ((0, 3), False),
                              ((2, 2), True), ((1, 3), True)):
            with self.subTest(sizes=sizes, static=static):
                torch.manual_seed(31)
                experts = torch.nn.Module()
                experts.local_expert_count = 2
                experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 6, 4) * 0.4)
                experts.down_proj = torch.nn.Parameter(torch.randn(2, 4, 3) * 0.4)
                reference = copy.deepcopy(experts)
                tokens = torch.randn(sum(sizes), 4, requires_grad=True)
                reference_tokens = tokens.detach().clone().requires_grad_()
                positions = (sizes[0], sum(sizes))
                offsets = positions if static else torch.tensor(positions)
                batch = ExpertBatch(tokens, offsets)
                actual = compute_expert_major(experts, batch)
                expected_parts = []
                start = 0
                for index, size in enumerate(sizes):
                    gate, up = F.linear(
                        reference_tokens[start:start + size], reference.gate_up_proj[index]
                    ).chunk(2, dim=-1)
                    expected_parts.append(F.linear(
                        F.silu(gate) * up, reference.down_proj[index]
                    ))
                    start += size
                expected = torch.cat(expected_parts)
                upstream = torch.randn_like(actual)
                (actual * upstream).sum().backward()
                (expected * upstream).sum().backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(tokens.grad, reference_tokens.grad, rtol=0, atol=0)
                for parameter, reference_parameter in zip(experts.parameters(), reference.parameters()):
                    torch.testing.assert_close(
                        parameter.grad, reference_parameter.grad, rtol=0, atol=0
                    )
                self.assertEqual(tuple(offsets), positions)

    def test_expert_major_batch_rejects_bad_dynamic_offsets(self):
        """Reject offsets that do not describe contiguous expert groups."""
        experts = torch.nn.Module()
        experts.local_expert_count = 2
        tokens = torch.randn(3, 4)
        for offsets in ((2, 2), (2, 1), (4, 3)):
            for representation in (tuple, torch.tensor):
                with self.subTest(offsets=offsets, representation=representation):
                    with self.assertRaisesRegex(ValueError, "offsets"):
                        compute_expert_major(experts, ExpertBatch(tokens, representation(offsets)))
        for offsets in ((True, 3), (1.5, 3)):
            with self.subTest(offsets=offsets):
                with self.assertRaisesRegex(ValueError, "offsets"):
                    compute_expert_major(experts, ExpertBatch(tokens, offsets))

    def test_routed_preflight_rejects_peer_layout_mismatch(self):
        """Check the full communication signature before all-to-all."""
        experts = _make_routed_experts()
        experts.num_experts = 4
        experts.local_expert_count = 2
        hidden = torch.randn(2, 8)
        indices = torch.zeros(2, 1, dtype=torch.long)
        weights = torch.ones(2, 1)
        for field, peer_value in ((1, 6), (2, 3), (3, 16), (4, "torch.bfloat16")):
            def gather_peer_plan(output: list, local_plan: tuple, group: object) -> None:
                """Return a peer signature with one incompatible layout field."""
                del group
                peer_signature = list(local_plan[0])
                peer_signature[field] = peer_value
                output[:] = [local_plan, (tuple(peer_signature), None)]

            with self.subTest(field=field):
                with (
                    patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
                    patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
                    patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                          side_effect=gather_peer_plan),
                    patch(_DISPATCHER_MODULE + ".fc.all_to_all_single_autograd") as collective,
                ):
                    with self.assertRaisesRegex(ValueError, "identical routing signatures"):
                        routed_forward(experts, hidden, indices, weights, object())
                    collective.assert_not_called()

    def test_graph_route_uses_bound_local_expert_count(self):
        """The eager binder need not add num_experts to the expert holder."""
        experts = torch.nn.Module()
        experts.local_expert_count = 2
        experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 12, 8))
        experts.down_proj = torch.nn.Parameter(torch.randn(2, 8, 6))
        hidden = torch.randn(2, 8)
        indices = torch.zeros(2, 1, dtype=torch.long)
        weights = torch.ones(2, 1)
        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                  side_effect=self._gather_local_plan),
            patch(_DISPATCHER_MODULE + ".ep_all_to_all",
                  side_effect=lambda rows, *_: rows) as exchange,
        ):
            output = routed_forward(experts, hidden, indices, weights, object())
            self.assertEqual(tuple(output.shape), (2, 8))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(exchange.call_count, 2)
            del experts.local_expert_count
            with self.assertRaisesRegex(ValueError, "local_expert_count"):
                routed_forward(experts, hidden, indices, weights, object())

    def test_eager_grouped_gemm_branch_stays_selected(self):
        """The shared SwiGLU helper must not replace the grouped kernel path."""
        experts = torch.nn.Module()
        experts.local_expert_count = 2
        experts._ep_use_grouped_gemm = True
        experts.forward_expert_major = unittest.mock.Mock(side_effect=lambda rows, counts: rows * 2)
        states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        indices = torch.tensor([1, 0, 0])
        with patch("hyper_parallel.distributed.expert_parallel.experts.swiglu_expert_forward") as helper:
            output = _local_swiglu_expert_forward(experts, states, indices)
        torch.testing.assert_close(output, states * 2)
        helper.assert_not_called()
        self.assertEqual(experts.forward_expert_major.call_count, 1)
        torch.testing.assert_close(
            experts.forward_expert_major.call_args.args[1], torch.tensor([2, 1])
        )

    def test_invalid_expert_ids_fail_before_collective(self):
        """Reject both negative and out-of-range IDs at the public route entry."""
        experts = torch.nn.Module()
        experts.num_experts = 4
        experts.local_expert_count = 2
        hidden = torch.randn(1, 2, 8)
        weights = torch.ones(2, 1)
        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                  side_effect=self._gather_local_plan),
            patch(_DISPATCHER_MODULE + ".fc.all_to_all_single_autograd") as collective,
        ):
            for invalid_id in (-1, 4):
                indices = torch.tensor([[0], [invalid_id]])
                with self.assertRaisesRegex(ValueError, "Top-K expert IDs"):
                    routed_forward(experts, hidden, indices, weights, object())
            collective.assert_not_called()

    def test_invalid_static_routing_never_enters_collective(self):
        """Reject incompatible expert shards and routing tensor shapes."""
        experts = torch.nn.Module()
        experts.num_experts = 8
        experts.local_expert_count = 4
        hidden = torch.randn(1, 3, 8)
        indices = torch.zeros(3, 2, dtype=torch.long)
        weights = torch.ones(3, 2)
        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                  side_effect=self._gather_local_plan),
            patch(_DISPATCHER_MODULE + ".fc.all_to_all_single_autograd") as collective,
        ):
            with self.assertRaisesRegex(ValueError, "matching Top-K"):
                routed_forward(experts, hidden, indices, weights[:, :1], object())
            with self.assertRaisesRegex(ValueError, "one row"):
                routed_forward(experts, hidden, indices[:2], weights[:2], object())
            experts.local_expert_count = 3
            with self.assertRaisesRegex(ValueError, "conflicts with the eager binder"):
                routed_forward(experts, hidden, indices, weights, object())
            collective.assert_not_called()

    def test_oversized_static_routing_fails_before_allocation(self):
        """Reject a dense dispatch beyond the validation budget on meta tensors."""
        experts = torch.nn.Module()
        experts.num_experts = 512
        experts.local_expert_count = 256
        hidden = torch.empty(1, 1024, 4096, device="meta", dtype=torch.bfloat16)
        indices = torch.empty(1024, 8, device="meta", dtype=torch.long)
        weights = torch.empty(1024, 8, device="meta", dtype=torch.float32)
        with (
            patch(_DISPATCHER_MODULE + ".dist.is_initialized", return_value=True),
            patch(_DISPATCHER_MODULE + ".dist.get_world_size", return_value=2),
            patch(_DISPATCHER_MODULE + ".dist.all_gather_object",
                  side_effect=self._gather_local_plan),
            patch(_DISPATCHER_MODULE + ".fc.all_to_all_single_autograd") as collective,
        ):
            with self.assertRaisesRegex(ValueError, "dense dispatch requires"):
                routed_forward(experts, hidden, indices, weights, object())
            collective.assert_not_called()
