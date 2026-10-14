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
"""Per-call dispatcher state, explicit selection and expert module hooks."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.compile.expert_parallel import graph_ep_trace
from hyper_parallel.distributed.expert_parallel.dispatch_contract import RoutingResult
from hyper_parallel.distributed.expert_parallel.dispatchers import FixedSlotTokenDispatcher
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    compute_expert_major,
    ep_routed_forward,
    ep_routed_dispatch,
    validate_expert_layout,
)


def _model(group, backend="fixed_slot"):
    """Bind a complete logical expert shard before FSDP hooks are installed."""
    model = torch.nn.Module()
    model.num_experts = 4
    model.experts = torch.nn.Module()
    model.experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 12, 8))
    model.experts.down_proj = torch.nn.Parameter(torch.randn(2, 8, 6))
    bind_local_expert_forward(model, 2, dispatcher_backend=backend, ep_group=group)
    return model


def _router(_module, hidden):
    """Use deterministic routes that leave several global experts empty."""
    count = hidden.numel() // hidden.shape[-1]
    return torch.zeros(count, 1, dtype=torch.long), torch.ones(count, 1)


@contextmanager
def _local_ep():
    """Mirror peer metadata and leave data rows unchanged in CPU-only tests."""
    def gather(output: list, plan: tuple, group: object) -> None:
        """Present the same local metadata on both simulated ranks."""
        del group
        output[:] = [plan, plan]

    prefix = "hyper_parallel.distributed.expert_parallel.dispatchers"
    with (
        patch(prefix + ".dist.is_initialized", return_value=True),
        patch(prefix + ".dist.get_world_size", return_value=2),
        patch(prefix + ".dist.all_gather_object", side_effect=gather),
        patch(prefix + ".ep_all_to_all", side_effect=lambda rows, *_: rows),
    ):
        yield


class TestTokenDispatcher(unittest.TestCase):
    """The selected backend and module boundary are independent of capture."""

    def test_capture_keeps_dispatch_compute_combine_and_hooks(self):
        """Observe identical implementation calls through both capture scopes."""
        group, events = object(), []
        model = _model(group)
        dispatcher = model.experts.token_dispatcher
        before = model.experts.register_forward_pre_hook(lambda *_: events.append("before"))
        after = model.experts.register_forward_hook(lambda *_: events.append("after"))
        hidden = torch.randn(3, 8)
        prefix = "hyper_parallel.distributed.expert_parallel.experts"
        with _local_ep(), patch(prefix + ".compute_expert_major", wraps=compute_expert_major) as compute:
            eager = ep_routed_forward(model, hidden, router_fn=_router, ep_group=group)
            with graph_ep_trace(True, model) as capture:
                captured = ep_routed_forward(model, hidden, router_fn=_router, ep_group=group)
            self.assertEqual(compute.call_count, 2)
        before.remove()
        after.remove()
        torch.testing.assert_close(eager, captured, rtol=0, atol=0)
        self.assertIs(model.experts.token_dispatcher, dispatcher)
        self.assertEqual(events, ["before", "after", "before", "after"])
        self.assertEqual(capture.groups, [group])
        self.assertEqual(capture.expert_param_names, {"experts.gate_up_proj", "experts.down_proj"})

    def test_metadata_can_be_combined_out_of_order(self):
        """A later dispatch cannot overwrite the first invocation's weights."""
        dispatcher = FixedSlotTokenDispatcher(object(), 2)
        hidden = torch.randn(3, 8)
        indices = torch.zeros(3, 1, dtype=torch.long)
        with _local_ep():
            first_plan = dispatcher.prepare(hidden, RoutingResult(indices, torch.ones(3, 1)))
            second_plan = dispatcher.prepare(hidden, RoutingResult(indices, torch.full((3, 1), 2.0)))
            first = dispatcher.dispatch(hidden, first_plan)
            second = dispatcher.dispatch(hidden, second_plan)
            latter = dispatcher.combine(second.batch.tokens, second.metadata)
            former = dispatcher.combine(first.batch.tokens, first.metadata)
        torch.testing.assert_close(latter, former * 2, rtol=0, atol=0)
        self.assertIsNot(first.metadata, second.metadata)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in vars(dispatcher).values()))

    def test_logical_shape_validation_precedes_materialization_hook(self):
        """Do not mistake a managed runtime shard for the logical expert layout."""
        group = object()
        model = _model(group)
        experts = model.experts
        full_gate = torch.clone(torch.detach(experts.gate_up_proj))
        full_down = torch.clone(torch.detach(experts.down_proj))
        experts.gate_up_proj.data = full_gate[:1]
        experts.down_proj.data = full_down[:1]
        experts.hsdp_scheduler = object()
        seen = []

        def materialize(module: torch.nn.Module, _inputs: tuple) -> None:
            """Simulate FSDP's pre-forward parameter materialization."""
            seen.append(tuple(module.gate_up_proj.shape))
            module.gate_up_proj.data = full_gate
            module.down_proj.data = full_down

        handle = experts.register_forward_pre_hook(materialize)
        with _local_ep():
            output = ep_routed_forward(model, torch.randn(3, 8), router_fn=_router, ep_group=group)
        handle.remove()
        self.assertEqual(seen, [(1, 12, 8)])
        self.assertEqual(tuple(output.shape), (3, 8))

    def test_dynamic_backend_is_not_replaced_during_capture(self):
        """An unadapted backend fails through group preflight without data exchange."""
        group = object()
        model = _model(group, "dynamic")
        with _local_ep(), graph_ep_trace(True, model):
            with self.assertRaisesRegex(ValueError, "dispatcher_backend='fixed_slot'"):
                ep_routed_forward(model, torch.ones(3, 8), router_fn=_router, ep_group=group)
        self.assertIsNone(model.experts.token_dispatcher)

    def test_expert_major_call_requires_explicit_offsets(self):
        """Indices and offsets cannot be inferred from positional tensor shapes."""
        model = _model(object())
        with self.assertRaisesRegex(ValueError, "either expert indices"):
            model.experts(torch.ones(3, 8), torch.zeros(3, dtype=torch.long), offsets=(1, 3))

    def test_fixed_eager_rejects_split_dispatch_before_token_exchange(self):
        """Explicit fixed selection cannot fall back to the legacy split algorithm."""
        group = object()
        model = _model(group)
        with _local_ep():
            with self.assertRaisesRegex(ValueError, "rank 0.*fixed_slot.*split dispatch"):
                ep_routed_dispatch(model, torch.ones(3, 8), router_fn=_router, ep_group=group)

    def test_local_mode_has_no_host_validation_and_same_math(self):
        """Local eager mode preserves output while avoiding diagnostic host sync."""
        group = object()
        model = _model(group)
        hidden = torch.randn(3, 8)
        with _local_ep():
            expected = ep_routed_forward(model, hidden, router_fn=_router, ep_group=group)
            model.experts.token_dispatcher = FixedSlotTokenDispatcher(group, 2, "local")
            with (
                patch("torch.Tensor.item", side_effect=AssertionError("Host scalar read")),
                patch("torch.distributed.all_gather_object", side_effect=AssertionError("Host collective")),
            ):
                actual = ep_routed_forward(model, hidden, router_fn=_router, ep_group=group)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_local_mode_still_agrees_during_capture(self):
        """Opting out of eager diagnostics must not bypass capture agreement."""
        group = object()
        model = _model(group)
        model.experts.token_dispatcher = FixedSlotTokenDispatcher(group, 2, "local")
        with _local_ep(), graph_ep_trace(True, model):
            with patch("torch.distributed.all_gather_object", side_effect=RuntimeError("Peer gate")):
                with self.assertRaisesRegex(RuntimeError, "Peer gate"):
                    ep_routed_forward(model, torch.ones(3, 8), router_fn=_router, ep_group=group)

    def test_static_tensor_contracts_reject_before_data_exchange(self):
        """Device and dtype failures enter the same strict preparation gate."""
        hidden = torch.randn(3, 8)
        routes = [
            (torch.zeros(3, 1, dtype=torch.long, device="meta"), torch.ones(3, 1), "same device"),
            (torch.zeros(3, 1, dtype=torch.long), torch.ones(3, 1, device="meta"), "same device"),
            (torch.zeros(3, 1, dtype=torch.long), torch.ones(3, 1, dtype=torch.int64), "floating-point"),
        ]
        for indices, weights, reason in routes:
            group = object()
            model = _model(group)
            with self.subTest(reason=reason), _local_ep():
                with self.assertRaisesRegex(ValueError, "rank 0.*" + reason):
                    ep_routed_forward(model, hidden, router_fn=lambda *_: (indices, weights), ep_group=group)

    def test_expert_dtype_and_device_contracts(self):
        """Reject a mismatched matrix operand before expert compute."""
        model = _model(object())
        hidden = torch.ones(3, 8)
        model.experts.down_proj = torch.nn.Parameter(model.experts.down_proj.to(torch.float64))
        with self.assertRaisesRegex(ValueError, "compute dtype"):
            validate_expert_layout(model.experts, hidden)
        model.experts.down_proj = torch.nn.Parameter(torch.empty(2, 8, 6, device="meta"))
        with self.assertRaisesRegex(ValueError, "compute device"):
            validate_expert_layout(model.experts, hidden)

    def test_autocast_and_declared_fsdp_cast_are_supported(self):
        """Validate compute dtypes rather than requiring storage dtype equality."""
        model = _model(object())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            validate_expert_layout(model.experts, torch.ones(3, 8, dtype=torch.bfloat16))
        policy = SimpleNamespace(param_dtype=torch.bfloat16, cast_forward_inputs=True, output_dtype=None)
        model.experts.hsdp_scheduler = SimpleNamespace(mp_policy=policy, device=torch.device("cpu:0"))
        validate_expert_layout(model.experts, torch.ones(3, 8, dtype=torch.bfloat16))

    def test_unknown_validation_mode_is_rejected(self):
        """A misspelled policy cannot silently disable peer agreement."""
        with self.assertRaisesRegex(ValueError, "validation_mode"):
            FixedSlotTokenDispatcher(object(), 2, "typo")
