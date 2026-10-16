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
"""
Graph Compiler - compile a model into a parallel graph and run fwd+bwd.

Compiler surface of the graph-mode stack:

- ``compile`` traces the model into a joint forward+backward FX graph and
  runs the partitioning passes (FSDP / PP / overlap) on it.
- ``forward_backward`` executes the compiled graph against the live model
  state and deposits gradients into ``param.grad`` (accumulating, so several
  micro-batch calls may run before the caller's optimizer step).

Training-policy concerns (optimizer, dataloader loop, logging, batch device
placement) are deliberately out of scope: ``GraphTrainer`` composes this
class and owns them.
"""

__all__ = ["GraphCompiler"]

import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _register_process_group

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

from .pass_config import PassConfig, normalize_dp_mode, build_pass_config_from_trainer_config
from .expert_parallel import graph_ep_trace
from .graph_parallel_plan import GraphParallelPlan
from .passes.pipeline import PassPipeline
from .tracer.graph_tracer import run_traced_graph, trace_model_graph

_LOG = logging.getLogger(__name__)


class GraphCompiler:
    """
    Graph-mode Compiler

    Users provide model code and parallel configuration.
    The compiler handles graph capture and all parallel logic
    (FSDP / PP / overlap passes), then runs forward+backward per step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_fn: Callable,
        pass_config: Optional[PassConfig] = None,
        parallel_plan: Optional[GraphParallelPlan] = None,
        trainer_config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        mesh_context: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model: Model to compile
            train_fn: Training function signature: (model, input, label) -> loss
            pass_config: Parallel configuration. When omitted, the compiler
                projects ``trainer_config`` onto a graph-mode ``PassConfig``;
                if both are omitted, it uses ``PassConfig()``.
            parallel_plan: GraphParallelPlan declaring which modules to shard
                (optional; enables declarative sharding)
            trainer_config: Optional trainer topology used only to infer a
                default ``PassConfig`` when ``pass_config`` is omitted.
            device: Device to place the model and run the graph on. Defaults
                to the NPU device when available, otherwise CPU.
            mesh_context: Optional automodel ``MeshContext`` carrying a
                pre-built TP/FSDP mesh. When provided, the TP group is reused
                as-is (boundary forwards already hold the group object) and
                only the FSDP shard sub-mesh is registered under ``"fsdp"``.
                Use this to feed an automodel TP-sharded model into the
                graph-mode FSDP pass.
        """
        self.model = model
        self.train_fn = train_fn
        self.parallel_plan = parallel_plan
        self.pass_config = self._resolve_pass_config(
            pass_config=pass_config,
            trainer_config=trainer_config,
        )
        self._mesh_context = mesh_context
        self.device = device or (
            torch.device("npu")
            if (hasattr(torch, "npu") and torch.npu.is_available())
            else torch.device("cpu")
        )

        self.pass_config.validate()

        self._joint_graph = None
        self._ep_sync_group = None
        self._pytree_pre_hook: Optional[Callable[[], None]] = None

    @staticmethod
    def _resolve_pass_config(
        *,
        pass_config: Optional[PassConfig],
        trainer_config: Optional[Any],
    ) -> PassConfig:
        """Resolve pass configuration from explicit or trainer-level input."""
        if pass_config is not None:
            return pass_config
        if trainer_config is not None:
            return build_pass_config_from_trainer_config(trainer_config)
        return PassConfig()

    @property
    def is_compiled(self) -> bool:
        """Whether a joint graph has already been compiled."""
        return self._joint_graph is not None

    def compile(self, **inputs: Any) -> None:
        """
        Compile model into parallel graph

        Users can explicitly call this, or it will be automatically compiled
        at first forward_backward

        Args:
            **inputs: Model inputs, forwarded to ``train_fn`` as keyword
                arguments and used to trace the joint graph
        """
        if self._pytree_pre_hook is not None:
            self._pytree_pre_hook()

        if self.pass_config.ep_enabled:
            self._validate_ep_mesh()
        if self.pass_config.fsdp_enabled and dist.is_initialized():
            if self.pass_config.ep_enabled:
                self._validate_fsdp_ep_mesh()
            # Only build the FSDP mesh when distributed is actually up.
            # ``FSDPPass`` early-returns when ``world_size == 1``, so a
            # single-process run (no dist, or a single rank) compiles and
            # runs as plain graph mode without sharding.
            self._init_device_mesh(self._mesh_context)
            if self.pass_config.ep_enabled:
                self._init_expert_fsdp_group()

        with graph_ep_trace(self.pass_config.ep_enabled) as ep_groups:
            joint_graph = trace_model_graph(self.model, self.train_fn, inputs)
        if self.pass_config.ep_enabled:
            self._resolve_ep_sync_group(ep_groups)

        pipeline = PassPipeline.from_config(self.pass_config, self.parallel_plan)

        pass_kwargs = self._build_pass_kwargs()

        # Passes mutate ``graph_module`` in place and return it, so the
        # transformed graph lives on ``joint_graph`` for ``forward_backward``.
        pipeline.run(joint_graph.graph_module, **pass_kwargs)

        self._joint_graph = joint_graph

    def forward_backward(self, **inputs: Any) -> Any:
        """
        Execute one compiled forward+backward step.

        Compiles lazily on the first call. Gradients are ACCUMULATED into
        ``param.grad`` (not overwritten), so several micro-batch calls may
        run before the caller's optimizer step / zero_grad.

        Args:
            **inputs: Model inputs, forwarded to ``train_fn`` as keyword
                arguments (must live on the compiler's device)

        Returns:
            tuple: ``(loss, loss_dict)`` — the backward loss and the named
            loss values emitted by ``train_fn`` as extra graph outputs after
            the gradients (empty when ``train_fn`` returned a bare loss
            tensor).

        Note:
            EP loss validation synchronizes the captured EP group for pure
            EP, or the dense FSDP group for FSDP+EP. All ranks in that group
            must call this method in the same order. Independent pure EP
            groups may use different schedules. Captured EP groups must have
            identical membership for pure EP and be contained in the dense
            FSDP group for FSDP+EP; compilation checks these constraints.
        """
        if self._joint_graph is None:
            self.compile(**inputs)

        loss, grads, loss_dict = self._run_graph(**inputs)

        if self.pass_config.ep_enabled and dist.is_initialized():
            # Agree within this compiler's training participants; independent
            # EP subgroups need not execute steps in lockstep.
            finite = torch.isfinite(loss).all().to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=self._ep_sync_group)
            if not bool(finite.item()):
                raise FloatingPointError("Graph EP loss is nonfinite on at least one rank")

        self._accumulate_grads(grads)

        return loss, loss_dict

    def set_pytree_pre_hook(self, hook: Callable[[], None]) -> "GraphCompiler":
        """Register a no-arg hook run immediately before first compilation."""
        self._pytree_pre_hook = hook
        return self

    def to(self, device: torch.device) -> "GraphCompiler":
        """Move the model to ``device`` and remember it for graph execution."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def _init_device_mesh(self, mesh_context: Optional[Any] = None):
        """Initialize the DP process group(s) for the configured ``dp_mode``.

        Three modes (see ``PassConfig.dp_mode``), matching simplefsdp's
        ``data_parallel`` and the eager ``fully_shard`` conventions:

        * ``"fsdp"`` (default): a 1-D ``("fsdp_shard",)`` mesh; the group is
          registered under the graph-mode name ``"fsdp"``.
        * ``"ddp"``: a 1-D ``("fsdp_replicate",)`` mesh; registered under
          ``"dp_replicate"`` (no parameter sharding, grads all-reduced).
        * ``"hsdp"``: a 2-D ``("fsdp_replicate", "fsdp_shard")`` mesh
          (replicate, shard); both groups registered.

        An external automodel ``MeshContext`` is reused when provided — its
        ``fsdp_replicate`` / ``fsdp_shard`` sub-meshes are registered under the
        graph-mode names, mirroring the eager ``FSDP2Manager``. The fallback
        builds the mesh with the repo ``init_device_mesh`` (the same builder the
        eager stack uses) and only supports **pure DP** (``tp_size == 1``, PP
        off); TP/PP topologies must supply a ``MeshContext``.
        """
        dp_mode = normalize_dp_mode(self.pass_config.dp_mode)
        if mesh_context is not None:
            self._init_mesh_from_context(mesh_context, dp_mode)
            return

        self._require_pure_dp_fallback()
        device_type = (
            "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
        )
        world_size = dist.get_world_size()
        rank_list = tuple(range(world_size))

        if dp_mode == "ddp":
            mesh = init_device_mesh(
                device_type,
                (world_size,),
                mesh_dim_names=("fsdp_replicate",),
                rank_list=rank_list,
                init_backend=True,
            )
            self._register_mesh_group("dp_replicate", mesh["fsdp_replicate"])
            self.pass_config.dp_replicate_degree = mesh["fsdp_replicate"].size()
            return

        if dp_mode == "hsdp":
            shard, replicate = self._resolve_hsdp_degrees(world_size)
            mesh = init_device_mesh(
                device_type,
                (replicate, shard),
                mesh_dim_names=("fsdp_replicate", "fsdp_shard"),
                rank_list=rank_list,
                init_backend=True,
            )
            self._register_mesh_group("dp_replicate", mesh["fsdp_replicate"])
            self._register_mesh_group("fsdp", mesh["fsdp_shard"])
            self.pass_config.dp_replicate_degree = mesh["fsdp_replicate"].size()
            self.pass_config.fsdp_degree = mesh["fsdp_shard"].size()
            return

        # dp_mode == "fsdp" (default): 1-D shard mesh over the whole world.
        mesh = init_device_mesh(
            device_type,
            (world_size,),
            mesh_dim_names=("fsdp_shard",),
            rank_list=rank_list,
            init_backend=True,
        )
        self._register_mesh_group("fsdp", mesh["fsdp_shard"])
        # Back-fill, mirroring the external-mesh branch: FSDPPass resolves the
        # group size from ``fsdp_degree`` (falling back to world_size when
        # ``None``), so setting it here keeps the two paths consistent.
        self.pass_config.fsdp_degree = world_size

    def _require_pure_dp_fallback(self) -> None:
        """Reject TP/PP topologies for the self-built DP mesh.

        The fallback mesh spans the whole world; with a TP or PP axis it would
        shard/reduce across the wrong ranks. Those topologies must supply an
        automodel ``MeshContext`` (which carries the proper DP sub-mesh).

        Raises:
            ValueError: When ``tp_size != 1`` or ``pp_enabled``.
        """
        cfg = self.pass_config
        if cfg.tp_size != 1 or cfg.pp_enabled:
            raise ValueError(
                "Graph-mode DP without a MeshContext only supports pure data "
                f"parallelism (tp_size=1, PP disabled); got tp_size={cfg.tp_size}, "
                f"pp_enabled={cfg.pp_enabled}. Pass a MeshContext carrying the DP "
                "sub-mesh for TP/PP hybrids."
            )

    def _init_mesh_from_context(self, mesh_context: Any, dp_mode: str) -> None:
        """Resolve and register the DP group(s) from an automodel ``MeshContext``.

        Reuses automodel's already-built mesh (boundary forwards hold the TP
        group object directly) and registers the ``fsdp_replicate`` /
        ``fsdp_shard`` sub-meshes under the graph-mode names ``"dp_replicate"``
        / ``"fsdp"``. The effective mode is first reconciled with the mesh
        topology (see ``_reconcile_mode_with_context``), mirroring the eager
        ``FSDP2Manager`` which picks HSDP from ``dp_replicate_size > 1``.
        """
        fsdp_mesh = (
            getattr(mesh_context, "fsdp_non_moe_mesh", None) or mesh_context.device_mesh
        )
        names = tuple(getattr(fsdp_mesh, "mesh_dim_names", ()) or ())
        dp_mode = self._reconcile_mode_with_context(mesh_context, dp_mode)
        self.pass_config.dp_mode = dp_mode
        if self.pass_config.ep_enabled:
            self.pass_config.validate()

        if dp_mode == "ddp":
            dim = self._first_dim(names, ("fsdp_replicate", "dp_replicate", "dp"))
            if dim is None:
                raise ValueError(
                    "dp_mode='ddp' needs a replicate axis among "
                    "('fsdp_replicate', 'dp_replicate', 'dp') in the mesh "
                    f"dims {names}"
                )
            sub = fsdp_mesh[dim]
            self._register_mesh_group("dp_replicate", sub)
            self.pass_config.dp_replicate_degree = sub.size()
            return

        if dp_mode == "hsdp":
            rep_dim = self._first_dim(names, ("fsdp_replicate", "dp_replicate"))
            shard_dim = self._first_dim(names, ("fsdp_shard", "fsdp", "dp"))
            if rep_dim is None or shard_dim is None:
                raise ValueError(
                    "dp_mode='hsdp' needs both a replicate axis "
                    "('fsdp_replicate'/'dp_replicate') and a shard axis "
                    f"('fsdp_shard'/'fsdp'/'dp') in the mesh dims {names}"
                )
            rep_sub = fsdp_mesh[rep_dim]
            shard_sub = fsdp_mesh[shard_dim]
            self._register_mesh_group("dp_replicate", rep_sub)
            self._register_mesh_group("fsdp", shard_sub)
            self.pass_config.dp_replicate_degree = rep_sub.size()
            self.pass_config.fsdp_degree = shard_sub.size()
            return

        # dp_mode == "fsdp": automodel's fsdp_non_moe_mesh is
        # ("fsdp_replicate","fsdp_shard","tp"); device_mesh (cp=1) is
        # ("dp","cp","tp") and "dp" is the FSDP axis.
        dim = "fsdp_shard" if "fsdp_shard" in names else "dp"
        sub = fsdp_mesh[dim]
        self._ep_sync_group = self._register_mesh_group("fsdp", sub)
        self.pass_config.fsdp_degree = sub.size()

    def _resolve_ep_sync_group(self, ep_groups: list) -> None:
        """Choose one participant set covering every captured EP collective."""
        if not ep_groups:
            raise ValueError("Graph EP capture found no EP communication group")
        group = self._ep_sync_group if self.pass_config.fsdp_enabled else ep_groups[0]
        if group is None:
            raise ValueError("Graph FSDP+EP requires a registered dense FSDP group")
        participants = set(dist.get_process_group_ranks(group))
        for ep_group in ep_groups:
            ranks = set(dist.get_process_group_ranks(ep_group))
            if self.pass_config.fsdp_enabled:
                valid = ranks <= participants
            else:
                valid = ranks == participants
            if not valid:
                raise ValueError("Captured EP group is incompatible with the compiler's loss synchronization group")
        self._ep_sync_group = group

    def _validate_ep_mesh(self) -> None:
        """Reject unsupported actual axes in either pure EP or FSDP+EP."""
        if self._mesh_context is None:
            return
        for axis in ("tp_size", "pp_size", "cp_size"):
            size = getattr(self._mesh_context, axis, 1)
            if size != 1:
                raise ValueError(
                    f"Graph EP does not support MeshContext.{axis}={size}"
                )
        for mode in ("sequence_parallel", "loss_parallel"):
            if getattr(self._mesh_context, mode, False):
                raise ValueError(
                    f"Graph EP does not support MeshContext.{mode}"
                )

    def _validate_fsdp_ep_mesh(self) -> None:
        """Reject unsupported replica axes and missing expert shard meshes."""
        if self._mesh_context is None:
            raise ValueError("Graph FSDP+EP requires a MeshContext with fsdp_moe_mesh")
        if getattr(self._mesh_context, "dp_replicate_size", 1) > 1:
            raise ValueError("Graph FSDP+EP does not support dense HSDP replication")
        expert_mesh = getattr(self._mesh_context, "fsdp_moe_mesh", None)
        if expert_mesh is None or "edp_shard" not in expert_mesh.mesh_dim_names:
            raise ValueError("Graph FSDP+EP requires an edp_shard axis in fsdp_moe_mesh")
        if ("edp_replicate" in expert_mesh.mesh_dim_names
                and operator.getitem(expert_mesh, "edp_replicate").size() > 1):
            raise ValueError("Graph FSDP+EP does not support edp_replicate expert replicas")

    def _init_expert_fsdp_group(self) -> None:
        """Register the expert shard axis from the same mesh used by eager EP."""
        if self._mesh_context is None:
            raise ValueError("Graph FSDP+EP requires a MeshContext with fsdp_moe_mesh")
        expert_mesh = getattr(self._mesh_context, "fsdp_moe_mesh", None)
        if expert_mesh is None or "edp_shard" not in expert_mesh.mesh_dim_names:
            raise ValueError("Graph FSDP+EP requires an edp_shard axis in fsdp_moe_mesh")
        self._register_mesh_group("fsdp_expert", operator.getitem(expert_mesh, "edp_shard"))

    def _reconcile_mode_with_context(self, mesh_context: Any, dp_mode: str) -> str:
        """Align ``dp_mode`` with the ``MeshContext`` topology (eager parity).

        The eager ``fully_shard`` path chooses FSDP vs HSDP from the mesh
        (``dp_replicate_size > 1`` means HSDP), not from a config flag. Mirror
        that so a mesh carrying a replicate axis is never silently treated as
        plain FSDP, and a 1-wide replicate axis is treated as FSDP.

        Raises:
            ValueError: When ``dp_mode='ddp'`` but the mesh shards
                (``dp_shard_size > 1``).
        """
        rep = getattr(mesh_context, "dp_replicate_size", None)
        shard = getattr(mesh_context, "dp_shard_size", None)
        if not isinstance(rep, int):
            # Not a real MeshContext (e.g. a test stub): leave the mode as-is.
            return dp_mode
        if dp_mode == "fsdp" and rep > 1:
            _LOG.warning(
                "dp_mode='fsdp' but MeshContext.dp_replicate_size=%d > 1; "
                "using 'hsdp' to match the mesh",
                rep,
            )
            return "hsdp"
        if dp_mode == "hsdp" and rep <= 1:
            _LOG.warning(
                "dp_mode='hsdp' but MeshContext.dp_replicate_size=%d <= 1; "
                "using 'fsdp'",
                rep,
            )
            return "fsdp"
        if dp_mode == "ddp" and isinstance(shard, int) and shard > 1:
            raise ValueError(
                "dp_mode='ddp' requires a non-sharding mesh "
                f"(dp_shard_size=1), got dp_shard_size={shard}; use 'hsdp'"
            )
        return dp_mode

    @staticmethod
    def _first_dim(names: tuple, candidates: tuple) -> Optional[str]:
        """Return the first candidate axis present in ``names`` (or None)."""
        for cand in candidates:
            if cand in names:
                return cand
        return None

    @staticmethod
    def _register_mesh_group(name: str, sub_mesh: Any) -> dist.ProcessGroup:
        """Register ``sub_mesh``'s process group under ``name``."""
        group = sub_mesh.get_group()
        _register_process_group(name, group)
        return group

    def _resolve_hsdp_degrees(self, world_size: int) -> Tuple[int, int]:
        """Resolve ``(shard, replicate)`` for a fallback 2-D HSDP mesh.

        Either degree may be left ``None`` and is derived from ``world_size``
        and the other; both must be consistent with ``world_size``.

        Raises:
            ValueError: When both are ``None``, a degree does not divide
                ``world_size``, or the product does not equal ``world_size``.
        """
        shard = self.pass_config.fsdp_degree
        replicate = self.pass_config.dp_replicate_degree
        if shard is None and replicate is None:
            raise ValueError(
                "dp_mode='hsdp' requires fsdp_degree and/or "
                "dp_replicate_degree to build the 2-D (dp_replicate, fsdp) mesh"
            )
        if shard is None:
            if world_size % replicate != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by "
                    f"dp_replicate_degree ({replicate})"
                )
            shard = world_size // replicate
        if replicate is None:
            if world_size % shard != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by fsdp_degree ({shard})"
                )
            replicate = world_size // shard
        if shard * replicate != world_size:
            raise ValueError(
                f"hsdp degrees fsdp_degree={shard} x "
                f"dp_replicate_degree={replicate} != world_size={world_size}"
            )
        return shard, replicate

    def _build_pass_kwargs(self) -> dict:
        """
        Build kwargs to pass to passes.
        """
        kwargs = {}

        # Live model: partitioning passes (FSDPPass) physically shard
        # parameters in place, keeping the compiler FSDP-agnostic.
        kwargs["model"] = self.model

        if self.pass_config.fsdp_enabled:
            kwargs["fsdp_group_name"] = "fsdp"
            if self.pass_config.ep_enabled:
                expert_names = set()
                for module_name, module in self.model.named_modules():
                    if not hasattr(module, "local_expert_count"):
                        continue
                    for param_name, _ in module.named_parameters():
                        expert_names.add(f"{module_name}.{param_name}" if module_name else param_name)
                if not expert_names:
                    raise ValueError("Graph FSDP+EP found no local expert parameters")
                kwargs["expert_param_names"] = expert_names
                kwargs["expert_fsdp_group_name"] = "fsdp_expert"
            # DDP / HSDP also reduce gradients on the replicate axis.
            if normalize_dp_mode(self.pass_config.dp_mode) in ("ddp", "hsdp"):
                kwargs["dp_replicate_group_name"] = "dp_replicate"

        return kwargs

    def _run_graph(self, **inputs):
        """Execute compiled graph"""
        if self._joint_graph is None:
            raise RuntimeError(
                "Graph not compiled. Call compiler.compile() or "
                "compiler.forward_backward() first."
            )

        # The joint graph's parameters/buffers are static inputs: feed the
        # live (FSDP-sharded) model state in FQN order each step.
        return run_traced_graph(
            self._joint_graph,
            self.model,
            inputs,
        )

    def _accumulate_grads(self, grads: List[torch.Tensor]) -> None:
        """Accumulate graph-computed gradients into the live model's parameters.

        The graph emits gradients in ``state_fqns`` order (trainable
        parameters only, shared parameters included once per FQN), which
        diverges from ``model.parameters()`` (deduplicated) when the model
        ties weights. Mapping by FQN keeps every gradient on the right
        parameter; the count check refuses to assign on mismatch instead of
        letting ``zip`` silently truncate.

        Accumulation (not overwrite) keeps ``forward_backward`` composable:
        several micro-batch steps may run before the caller's optimizer step
        (which ends with ``zero_grad``), so per-step grads must sum into
        ``param.grad``.
        """
        # ``state_is_param`` is attached to the traced GraphModule by
        # ``trace_model_graph``, not to the JointGraph dataclass itself.
        state_is_param = getattr(self._joint_graph.graph_module, "state_is_param", None)
        fqn_to_param = dict(self.model.named_parameters(remove_duplicate=False))
        trainable = self._trainable_params_in_state_order(
            self._joint_graph.state_fqns, state_is_param, fqn_to_param
        )
        if len(trainable) != len(grads):
            raise ValueError(
                f"Gradient count ({len(grads)}) does not match trainable "
                f"parameter count ({len(trainable)}). The traced graph and "
                f"the live model disagree on which parameters are trainable; "
                f"refusing to assign gradients to avoid silent misalignment."
            )

        for param, grad in zip(trainable, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    @staticmethod
    def _trainable_params_in_state_order(
        state_fqns: List[str],
        state_is_param: Optional[List[bool]],
        fqn_to_param: Dict[str, torch.nn.Parameter],
    ) -> List[torch.nn.Parameter]:
        """Return the live trainable parameters in the graph's state order.

        Mirrors the tracer's gradient emission order (``state_fqns`` order,
        parameters only, trainable only, shared parameters kept per FQN), so
        gradient ``i`` belongs to the returned parameter ``i``. Buffers share
        ``state_fqns`` but are absent from the parameter lookup; they are
        skipped explicitly so a missing ``state_is_param`` flag (old traces)
        degrades to parameter-only instead of raising KeyError.
        """
        trainable: List[torch.nn.Parameter] = []
        for idx, fqn in enumerate(state_fqns):
            if state_is_param is not None and not state_is_param[idx]:
                continue
            param = fqn_to_param.get(fqn)
            if param is not None and param.requires_grad:
                trainable.append(param)
        return trainable
