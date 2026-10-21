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
from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.distributed.mesh import MeshContext

from hyper_parallel.compile.ep_collectives import ep_all_to_all
from hyper_parallel.compile.ep_compute import validate_expert_layout, compute_expert_major
from hyper_parallel.compile.ep_contract import ExpertBatch, RoutingResult
from hyper_parallel.compile.ep_dispatch import FixedSlotTokenDispatcher
from hyper_parallel.compile.expert_parallel import routed_forward, graph_ep_trace, _GRAPH_EP_CAPTURE


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



class TestStaticEPPreparation(unittest.TestCase):
    """Coordinate predictable local failures before data communication."""

    @staticmethod
    def _experts():
        """Build a valid local expert holder without touching native binders."""
        experts = torch.nn.Module()
        experts.local_expert_count = 2
        experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 12, 8))
        experts.down_proj = torch.nn.Parameter(torch.randn(2, 8, 6))
        return experts

    def test_compute_dtype_and_device_are_validated(self):
        """A local matmul mismatch must fail before the first all-to-all."""
        experts = self._experts()
        validate_expert_layout(experts, torch.randn(4, 8))
        with self.assertRaisesRegex(ValueError, 'dtype'):
            validate_expert_layout(experts, torch.randn(4, 8, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, 'device'):
            validate_expert_layout(experts, torch.empty(4, 8, device='meta'))

    def test_offsets_are_the_only_static_group_definition(self):
        """Honor irregular exclusive offsets and reject tensor offsets."""
        experts = self._experts()
        tokens = torch.randn(4, 8, requires_grad=True)
        result = compute_expert_major(experts, ExpertBatch(tokens, (1, 4)))
        expected = torch.cat([torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(chunk, experts.gate_up_proj[index])[:, :6])
            * torch.nn.functional.linear(chunk, experts.gate_up_proj[index])[:, 6:],
            experts.down_proj[index]) for index, chunk in enumerate((tokens[:1], tokens[1:]))])
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'tuple'):
            compute_expert_major(experts, ExpertBatch(tokens, torch.tensor([1, 4])))

    def test_local_failure_is_shared_before_token_exchange(self):
        """An expert dtype failure reaches peers before either data exchange."""
        experts = self._experts().to(dtype=torch.bfloat16)

        def gather(output: list, value: tuple, group: object) -> None:
            """Expose one local failure to the mocked peer group."""
            del group
            output[:] = [value, (None, None)]

        with (patch('hyper_parallel.compile.ep_dispatch.dist.is_initialized', return_value=True),
              patch('hyper_parallel.compile.ep_dispatch.dist.get_world_size', return_value=2),
              patch('hyper_parallel.compile.ep_dispatch.dist.all_gather_object', side_effect=gather),
              patch('hyper_parallel.compile.ep_dispatch.ep_all_to_all') as exchange):
            with self.assertRaisesRegex(ValueError, 'rank 0.*dtype'):
                routed_forward(experts, torch.ones(4, 8), torch.zeros(4, 2, dtype=torch.int64),
                               torch.ones(4, 2), object())
            exchange.assert_not_called()

    def test_capture_scope_restores_after_error(self):
        """Nested graph contexts never leak into later calls."""
        self.assertIsNone(_GRAPH_EP_CAPTURE.get())
        with graph_ep_trace(True) as outer:
            self.assertIs(_GRAPH_EP_CAPTURE.get(), outer)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with graph_ep_trace(True):
                    raise RuntimeError('injected')
            self.assertIs(_GRAPH_EP_CAPTURE.get(), outer)
        self.assertIsNone(_GRAPH_EP_CAPTURE.get())



class TestStaticEPCompiler(unittest.TestCase):
    """Reject unsupported capture configurations before tracing."""

    def test_dynamic_shape_rejected_before_tracing(self):
        """The first graph EP stage keeps fixed input shapes."""
        from_config = PassConfig(ep_enabled=True, fsdp_enabled=False)
        compiler = GraphCompiler(torch.nn.Linear(8, 8), lambda *_: None, from_config,
                                 device=torch.device('cpu'), dynamic=True)
        with patch('hyper_parallel.compile.compiler.trace_model_graph') as trace:
            with self.assertRaisesRegex(ValueError, 'requires static shapes'):
                compiler.compile(tokens=torch.ones(2, 8))
            trace.assert_not_called()

    def test_pass_config_positional_arguments_are_preserved(self):
        """The added graph EP option cannot change old positional meanings."""
        config = PassConfig(True, False, False)
        self.assertFalse(config.fsdp_reshard_after_forward)
        self.assertFalse(config.ep_enabled)


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
            for fsdp in (False,):
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
