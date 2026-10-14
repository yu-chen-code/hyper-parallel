# HyperParallel Graph Mode

Graph-mode architecture for automatic parallelization with FSDP.

> **Note**: The `compile/` subpackage relies on a patched autograd engine for
> joint-graph capture (see Limitations). Its `torch.*` imports — including
> `torch.fx.experimental` and `torch._guards` — are therefore intentional and
> carry a `forbidden-backend-import` suppression in
> `.jenkins/check/config/filter_pylint.txt`.

## Fixed-split expert parallelism

For a pure EP job, keep the existing token routing and local expert computation
in the model. Call `graph_ep_all_to_all` for dispatch and combine when the
per-peer split sizes are fixed for every graph replay:

```python
from hyper_parallel.compile import GraphCompiler, PassConfig, graph_ep_all_to_all

# Inside the model's forward, after token routing:
received = graph_ep_all_to_all(tokens, send_counts, recv_counts, ep_group)
# ... run this rank's local experts ...
restored = graph_ep_all_to_all(local_output, recv_counts, send_counts, ep_group)

compiler = GraphCompiler(model, train_fn, PassConfig(fsdp_enabled=False))
```

Each count list must contain one non-negative Python integer per EP rank;
`sum(send_counts)` must equal the input's leading dimension. The caller must
derive matching counts on every rank before capture. Data-dependent routing
that changes split sizes between steps needs a separate capacity or recompilation
policy; this fixed-split entry does not infer one. TP and PP are outside this
EP path.

## Core Concept

**User**: Write model code + parallel configuration
**Framework**: Graph capture → DP/PP partitioning → Communication-compute overlap → Execution

## Architecture

### Layer 1: Configuration

```python
from hyper_parallel.compile import GraphParallelPlan, PassConfig

# Configure which modules to mark for FSDP
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark("tok_embeddings")
parallel_plan.fsdp_mark_pattern("layers.*")

# Parallel configuration
pass_config = PassConfig(enable_overlap=True)
```

### Layer 2: Graph Capture

`trace_model_graph` captures forward + backward into a joint FX graph:

- Parameters/buffers are static inputs (placeholders), not `get_attr` nodes
- `torch.autograd.grad` runs inside the traced function
- Uses `FakeTensorMode` + `make_fx` for symbolic tracing

### Layer 3: Pass Pipeline

```text
DeadCodeElimination → CanonicalizeGraph → FSDPPass → PpPass → AutoOverlapPass
```

**FSDPPass** (data-parallel, `PassConfig.dp_mode`):

- Identifies DP parameter placeholders via GraphParallelPlan
- Sinks `all_gather` to each parameter's first forward use (Shard → Replicate)
- Frees the replicated parameter after its last forward read, then re-gathers
  it for the backward and rematerializes any saved forward view of it
  (`reshard_after_forward`), so peak memory tracks the forward working set
- Inserts gradient reduction on gradient outputs — `reduce_scatter`
  (Replicate → Shard) for `"fsdp"` / `"hsdp"`, `all_reduce` on the
  `dp_replicate` axis for `"ddp"` / `"hsdp"`
- Physically shards live model parameters (dim 0) so optimizer is FSDP-agnostic
  (`"ddp"` replicates instead; non-divisible dim-0 params stay replicated
  rather than erroring)

Disable reshard with `PassConfig(fsdp_reshard_after_forward=False)`.

**PpPass** (pipeline-parallel, `PassConfig.pp_enabled`):

- Splits the (post-FSDP) joint graph to this rank's stage at module-FQN
  boundaries, exchanging boundary activations/gradients via P2P
- Installs a self-contained GPipe / 1F1B schedule (`pp_schedule`) as a
  `call_module` stub, so the trainer needs no PP wiring
- v1 is pure-PP (`pp_degree == world_size`, `fsdp_enabled=False`); PP+FSDP
  hybrids require a `mesh_context` carrying a pp dim

**AutoOverlapPass**:

- **Placeholder no-op today**: `enable_overlap` is accepted, but the pass
  does not yet reorder `wait_tensor` nodes. Sinking waits past independent
  compute is planned.

### Layer 4: Execution

```python
from hyper_parallel.compile import GraphTrainer

trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    pass_config=pass_config,
    parallel_plan=parallel_plan,
)

# Compile on first batch, then run forward + backward + optimizer
trainer.train(dataloader, max_steps=100, log_interval=10)
```

The compiled graph is a `GraphModule` that:

- Takes sharded parameters as inputs
- Gathers them via AllGather at each step
- Computes forward + backward
- Scatters gradients via ReduceScatter
- Outputs loss + grads

## Usage

```python
from hyper_parallel.compile import GraphParallelPlan, GraphTrainer, PassConfig

# 1. Model
model = Llama3Model(config)

# 2. Parallel plan
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark_pattern("layers.*")

# 3. Trainer
trainer = GraphTrainer(
    model=model,
    train_fn=lambda m, x, y: m(x).loss(y),
    pass_config=PassConfig(enable_overlap=True),
    parallel_plan=parallel_plan,
)

# 4. Training
trainer.train(dataloader, max_steps=1000)
```

## Dynamic input shapes

Static tracing remains the default. Enable symbolic user inputs with one option:

```python
compiler = GraphCompiler(model=model, train_fn=train_fn, dynamic=True)
```

For precise control, pass `dynamic_arg_dims` to `GraphCompiler` or `GraphTrainer`:

```python
compiler = GraphCompiler(
    model=model,
    train_fn=lambda m, x, y: ((m(x) - y) ** 2).mean(),
    dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
)
# x: [batch, sequence, input_features]; y: [batch, sequence, output_features]
loss, loss_dict = compiler.forward_backward(x=x, y=y)
```

The first call captures one symbolic forward/backward graph. Later batches with
compatible batch/sequence lengths reuse it; gradients still accumulate into the
live parameters. `GraphTrainer` exposes the same options and owns optimizer steps.
Numerical and graph-reuse regressions live in
[the CPU unit tests](../../tests/ut/compile/test_dynamic_shapes.py).

For the existing **BaseTrainer + GraphCompiler** path, configure:

```yaml
compile:
  enabled: true
  use_joint_graph: true
  dynamic_arg_dims:
    model_inputs.input_ids: [0, 1]
    labels: [0, 1]
```

Add paths for other tensors whose sizes vary, such as
`model_inputs.attention_mask: [0, 1]`. Paths refer to the keyword arguments of
`train_fn`; BaseTrainer supplies `model_inputs` and `labels`. Use `dynamic: true`
instead of the mapping to automatically symbolize all user tensor dimensions.

- `dynamic_arg_dims` overrides `dynamic`; unlisted dimensions stay static.
  Nested input paths and negative dimension indices are supported.
- Model parameters and buffers keep static shapes. Runtime guards reject
  incompatible shapes, metadata and Python constants without recompiling.
- Use a representative first batch with dynamic dimensions **greater than 1**;
  sizes 0/1 and shape-dependent Python branches may specialize the graph.
- Dynamic shapes support joint forward/backward graphs, gradient accumulation
  and optimizer updates. Pipeline parallel and data-dependent output shapes
  are not supported.

The approach follows MagiCompiler's separation of dynamic and static dimensions
and reuse of a general symbolic graph, adapted to HyperParallel's joint-graph
tracer. It adds no runtime dependency on the separate MagiCompiler checkout.

## Key Design Decisions

1. **Static Inputs**: Parameters are graph inputs, not `get_attr`. This allows passes to split the graph by reshaping placeholders.

2. **Joint Graph**: Forward + backward captured together via `torch.autograd.grad` inside the traced function.

3. **FSDP-Agnostic Optimizer**: FSDPPass shards the live model's parameters in place. `model.parameters()` returns shards, so optimizer needs no FSDP awareness.

4. **Declarative Sharding**: GraphParallelPlan uses FQN patterns (`layers.*`) instead of imperative module wrapping.

## Parallelism support

| Axis | Status | Notes |
|------|--------|-------|
| FSDP / DDP / HSDP | Implemented | `FSDPPass`; `PassConfig.dp_mode` selects the mode |
| PP | Implemented | `PpPass` + `pp_schedule` (`gpipe` / `1f1b`); pure-PP v1 |
| TP (+ SP / LP) | Via `mesh_context` | TP collectives are baked into automodel boundary forwards; graph mode reuses the mesh |
| EP | Static routed SwiGLU path | Fixed token shape and padded expert slots; pure EP or FSDP+EP |

For fixed-shape routed MoE, explicitly select `dispatcher_backend="fixed_slot"`
when binding the model, for both eager and graph execution. The generic
`build_ep_compute` and DeepSeek-V4.1 factory accept this keyword; the binder
also accepts it together with the actual `ep_group`. Preserve the model
factory's activation/custom gate configuration. Then compile the same forward
with `PassConfig(fsdp_enabled=False, ep_enabled=True)` (or enable FSDP below).
`GraphCompiler` does not replace a model's dispatcher. The default `dynamic`
backend keeps ragged eager routing and is rejected during capture preflight.
The learned Router and the recipe's shared-expert merge remain model-owned.

The lower-level `graph_ep_routed_forward(experts, hidden_states, topk_indices,
topk_weights, ep_group)` remains available for explicit graph adapters. Both
paths reuse the same stacked expert weights and functional all-to-all, with one
slot per local token and Top-K position for every expert. Replaying with
another route selection needs no new split plan; changing the number of tokens
requires recompilation. Graph EP rejects `dynamic=True` and explicit
`dynamic_arg_dims` before tracing; the general compiler's symbolic shape
support does not yet cover fixed-slot EP.
Fixed-slot eager and graph execution use the same `FixedSlotTokenDispatcher`
and the same `ep_routed_forward` pipeline: Router and local preparation,
group preflight, `dispatch`, `experts(tokens, offsets=...)`, then `combine`.
The expert module call preserves FSDP pre/post hooks. Weights are resolved
inside compute after hooks have materialized them; no parameter tensors are
cached in the dispatcher or plan. Both modes use one
`swiglu_expert_forward` primitive for fused or separate weights, the configured
activation or custom gate, and the down projection. Eager grouped GEMM remains
an optional specialized kernel of the separate dynamic backend.
Both non-grouped paths now pass expert-major rows through `ExpertBatch(tokens,
offsets)` and `compute_expert_major`. Offsets are exclusive end positions: eager
uses the actual per-expert counts, while graph permutes the received rank-major
rows and uses fixed per-expert capacity. The graph restores rank-major order
before combine. Eager reads counts once and computes their prefix sum on the
host, without an extra device `cumsum`. Fixed offsets are supplied as a tuple of Python shape values;
they are the only grouping boundaries during tracing and replay does not read
an offset tensor on the host. Graph offsets
describe capacity, including padded rows; they do not identify valid tokens.
The current bias-free SwiGLU path still requires zero input to produce zero
output. Explicit validity metadata belongs to a later dispatcher interface.
Eager keeps the `module.experts(...)` boundary for FSDP hooks and its async,
chunked and grouped-GEMM dispatch options. Graph routing derives the global
expert count from the eager binder's `local_expert_count * ep_size`; the expert
holder need not expose `num_experts`. If it does, the two counts must agree.
Graph capture rejects the eager split-dispatch path used by
`overlap_shared_expert=True`; that schedule needs a separate graph adapter.
One scoped `GraphEPCaptureContext` owns peer preflight records, captured groups,
and the exact expert weight names. It does not execute or select a route.
Router exceptions, invalid
local metadata, unsupported split dispatch, and invalid stacked SwiGLU weight
layouts all enter the same EP metadata agreement before either data all-to-all.
Weight validation checks rank, local expert count, compatible input/gate/up/output
dimensions, and compute devices/dtypes. Hidden states, Router indices and weights
must share one device; hidden states and Router weights must be floating point.
Expert linear operands must have compatible dtypes, accounting for active
autocast and declared FSDP parameter/input casts. Peer signatures also include
the device type and expert output dtype; local device indices may differ.
Binding records logical shape metadata for
managed FSDP experts, so runtime shards are not mistaken for logical weights.
Compute resolves the currently materialized weights independently.
The low-level exchange function only exchanges tokens; group registration is
performed after successful preflight. Nested scopes restore the parent context.
At each routed call, all EP members must enter the normal or split entry with
the same EP group.
Every EP rank must have the same `T * K` slot count, global and local expert
counts, hidden width, and hidden dtype at capture. A metadata collective shares
local validation results and these layout fields
before all-to-all; one rank's invalid shape, expert ID, or dispatch budget
raises `ValueError` on every participating rank. Padding costs
memory and compute and is intended for correctness validation, not performance.
The fixed-slot backend accepts `validation_mode="strict"` (the safe default)
or `validation_mode="local"` in its binder and factories. Strict eager execution
reads concrete IDs and agrees on metadata before each call. This diagnostic
mode intentionally synchronizes device/host and is an exception to the usual
training hot-path rule. Use it for correctness validation and configuration
debugging, not eager throughput measurements.

After validating a homogeneous, static configuration, `local` eager mode runs
metadata-only checks without `.item()` or a metadata collective. Invalid IDs
still mark the output nonfinite on device. The caller owns rank-consistent
configuration and failure handling in this mode: a unilateral preparation error
raises locally and is not guaranteed to release other ranks. Capture always
uses strict peer preflight, even when the selected mode is `local`; captured
graphs contain only token collectives and tensor math. Neither mode changes
the selected dispatcher or its dispatch/compute/combine algorithm.
EP cannot currently be combined with PP, TP, sequence parallelism or loss
parallelism in graph mode.

`topk_indices` must contain integer expert IDs in `[0, num_experts)`. A direct
strict call with concrete indices raises `ValueError` before communication when this
contract is violated. During graph capture the indices are FakeTensors, so the
captured graph uses device tensor operations to mark an invalid replay output
as NaN. Before accumulating gradients or allowing an optimizer step,
`GraphCompiler.forward_backward` all-reduces a finite-loss flag across its
training participants and raises `FloatingPointError` on that group if any
member failed. Pure EP uses the actual group recorded during graph capture;
all captured EP collectives must have identical rank membership. Independent
pure EP groups may capture and step at different times. FSDP+EP uses the
registered dense FSDP group, which must contain every captured EP group.
Compilation checks these membership constraints without a WORLD collective.
Members of the selected group must call `forward_backward` in the same order.
This safety check adds one device collective and host decision per EP step.
Callers using the lower-level `graph_ep_routed_forward` directly must handle
nonfinite outputs consistently across their participating ranks.

### Real one-layer V4.1 acceptance

The checked-in `tests/torch/expert_parallel/test_v41_graph_ep.py` launchers run
the actual `DeepseekV41ForCausalLM`, not the tiny routed test model. Local
fixtures retain one decoder, hidden size 640, eight routed experts, Top-6,
learned Router, native clamped SwiGLU and the shared expert. Engram, vision and
the indexer are disabled in this validation crop; no tokenizer or weight
download is required. The fixture config matches the earlier task validation;
its inactive Engram asset omits the unused tokenizer mapping.

```bash
pytest -q -s tests/torch/expert_parallel/test_v41_graph_ep.py::test_v41_one_layer_ep
pytest -q -s tests/torch/expert_parallel/test_v41_graph_ep.py::test_v41_one_layer_fsdp_ep
pytest -q -s tests/torch/expert_parallel/test_v41_graph_ep.py::test_v41_one_layer_ep_fp32
pytest -q -s tests/torch/expert_parallel/test_v41_graph_ep.py::test_v41_one_layer_fsdp_ep_fp32
```

These launch two or four ranks, load identical initial weights
into dynamic eager, fixed eager and fixed graph models, and compare ten
FP32 updates, or the existing two BF16 updates, of LM loss, gradients and updated
parameters. Router/shared hooks verify
that those real branches execute. FSDP graph parameters are physically sharded;
the independent eager references average gradients explicitly. The separate
`test_fixed_slot_eager_fsdp_hooks` ST verifies eager FSDP materialization hooks.
Full-model BF16 comparisons retain `atol=rtol=0.03`; the tiny FP32/BF16 tests
remain the more sensitive core arithmetic checks. These are static-shape
correctness tests, not a performance or production-scale acceptance claim.
Ten-step BF16 full-model agreement has not been established: extending the
independent dynamic/fixed eager chains exceeded the existing embedding-gradient
tolerance at the third update. The threshold was not relaxed. Two BF16 updates
and ten FP32 updates are distinct validation contracts, not interchangeable
evidence of BF16 long-run agreement.

For FSDP+EP, pass the already-built eager `MeshContext` to `GraphCompiler` and
use the same EP submesh when binding the model's routed forward. For example, a
four-rank `(edp_shard=2, ep=2)` topology uses `dp_shard_size=4`, `ep_size=2`,
and `edp_shard_size=2`. Set `PassConfig(fsdp_enabled=True, ep_enabled=True)`.
The compiler registers the dense `fsdp_shard` group and expert `edp_shard`
group separately. Capture identifies only the resolved gate/up/down parameters
actually consumed by expert compute, excluding auxiliary holder submodules.
`FSDPPass` builds one `ParamShardSpec` plan containing the name, group, degree,
and shard/reduce decisions. FX transformation and live parameter sharding use
that same plan. It all-gathers and reduce-scatters expert
parameters on `edp_shard`, and shards other trainable parameters on the dense
FSDP group. Expert parameters with `edp_shard_size=1` remain local and are not
reduced. Only `dp_mode="fsdp"` is supported for the EP combination. The
compiler rejects a trainable expert parameter whose first dimension cannot
be sharded by `edp_shard_size`: leaving such a parameter replicated would
skip its gradient synchronization. Dense non-divisible parameters retain the
existing replicated behavior described below. The
compiler rejects both `edp_replicate_size > 1` and `dp_replicate_size > 1`
before tracing: expert replica gradient reduction and dense HSDP+EP are not
implemented. The actual `MeshContext` must also have `tp_size=pp_size=cp_size=1`
with sequence and loss parallelism disabled; these axes cannot be hidden by
default values in `PassConfig`. The same TP/PP/CP/SP/LP checks apply to pure EP
whenever a `MeshContext` is supplied. Effective DP mode is revalidated after mesh
reconciliation.

Before allocating dense slots, the routed path checks the per-rank dispatch
tensor size `E * T * K * H * hidden_element_bytes` and rejects values above
256 MiB. This is a validation-scale limit for **one tensor**, not a guarantee
that the full forward/backward fits device memory. The verified one-layer V4.1
case has `E=8`, `T=16`, `K=6`, `H=640`, and BF16 hidden states: its dispatch
tensor is 0.9375 MiB per rank. Router weights are cast to the routed hidden dtype
before the final weighted aggregation, matching eager EP when the router
computes in FP32 and hidden states are BF16.

### Shared dispatcher contract

The shared EP directory owns `RoutingResult`, `DispatchPlan`, `ExpertBatch`,
`DispatchMetadata`, `DispatchResult`, and the small `TokenDispatcher` Protocol.
`prepare(hidden, routing)` performs no token exchange. `dispatch` returns
expert-major rows and call-specific combine metadata. `combine` consumes that
metadata; the dispatcher retains only its group and static expert count.
Outstanding calls therefore do not overwrite each other's routing tensors.
Offsets are the sole expert grouping boundaries. The old positional
`experts(states, local_expert_indices)` call remains supported, while
expert-major callers must use the explicit `offsets=` keyword.

The public `graph_ep_routed_forward` is a compatibility wrapper that explicitly
requests fixed slots and delegates to this same pipeline. Public contracts
contain tensor/shape metadata and have no FX or GraphCompiler dependency.
The shared implementation never imports Compile. Fixed-capacity execution is
fully shared between eager and graph; capturing the dynamic ragged backend,
grouped GEMM, chunk overlap and dynamic input shapes remain future work.

## Limitations

- **Non-divisible dense parameters are skipped**: a dense parameter whose dim 0
  is not divisible by the DP shard degree (or a scalar parameter) is left
  replicated on both the graph and the live model. Trainable Expert parameters
  that cannot shard across `edp_shard` are rejected before pass transformation.
- **PP v1 is pure-PP**: `pp_degree == world_size` and `fsdp_enabled=False`.
  PP+FSDP hybrids require a `mesh_context` exposing a pp dim.
- **`AutoOverlapPass` is a placeholder no-op** — see the pass note above.
- **Compilation is not idempotent**: the partitioning passes mutate the live
  model in place, so `GraphCompiler.compile` may only be called once — a
  second call silently re-traces the already-partitioned model and yields an
  incorrect graph (it does not raise; guard the call site yourself).
- **Requires torch with a patched autograd engine** for joint-graph capture.
  On stock torch the backward half is annotated structurally instead (see
  `_annotate_autograd_backward` in `tracer/graph_tracer.py`).
