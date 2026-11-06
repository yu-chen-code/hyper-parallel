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
"""Bridge graph-owned parameter storage to the existing DCP chunk contract."""

from __future__ import annotations

from copy import copy
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist

from hyper_parallel.core.distributed_checkpoint.metadata import CHUNK_INFO, ChunkInfo, ChunkStorageMetadata
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout, infer_slice_area_by_layout

if TYPE_CHECKING:
    from .compiler import GraphCompiler


class GraphCheckpointAdapter:
    """Describe local graph tensors without replacing live Parameters or the DCP planner.

    Checkpoints are taken at optimizer-step boundaries after storage preparation.
    Model tensors, FQN-keyed optimizer moments and FP32 main weights share one
    chunk description. Scalar optimizer counters remain replicated.
    """

    def __init__(self, compiler: GraphCompiler) -> None:
        """Bind prepared storage and its canonical parameter chunk descriptions."""
        compiler.prepare_model()
        self.compiler = compiler
        self.parameters = dict(compiler.model.named_parameters(remove_duplicate=False))
        self.parameter_shapes = compiler.parameter_shapes
        self.parameter_layouts = compiler.parameter_layouts
        self.canonical_names = {id(parameter): name for name, parameter in compiler.model.named_parameters()}
        self.chunks = {
            name: self._parameter_chunk(self.canonical_names[id(parameter)], parameter)
            for name, parameter in self.parameters.items()
        }

    def _parameter_chunk(self, name: str, parameter: torch.nn.Parameter) -> ChunkInfo:
        """Locate the valid rectangle within source-first shard geometry."""
        logical_shape = self.parameter_shapes[name]
        source = self.compiler.source_layout_for_parameter(name)
        global_shape = logical_shape
        offsets = [0] * len(logical_shape)
        if source is not None and source[1] is not None:
            source_layout = Layout.from_device_mesh(source[1])
            source_layout.set_placements(tuple(source[0]))
            source_layout.placement_to_tensor_map(len(logical_shape))
            global_shape = tuple(source_layout.get_global_shape(logical_shape))
            rank = source_layout.rank_list.index(source_layout.mesh.rank)
            area = infer_slice_area_by_layout(source_layout, rank, global_shape)
            if tuple(end - start for start, end in area) != logical_shape:
                raise ValueError(f"Graph checkpoint requires a rectangular source shard for {name}")
            offsets = [start for start, _ in area]

        sizes = list(logical_shape)
        if name in self.compiler.sharded_parameter_names:
            mesh, axis = self._shard_axis(parameter)
            local_start = min(mesh.get_local_rank(axis) * parameter.shape[0], logical_shape[0])
            offsets[0] += local_start
            sizes[0] = min(parameter.shape[0], logical_shape[0] - local_start)
        return ChunkInfo(ChunkStorageMetadata(tuple(offsets), tuple(sizes)), tuple(global_shape))

    def _shard_axis(self, parameter: torch.nn.Parameter) -> tuple:
        """Resolve the physical FSDP axis from the compiler's final Layout."""
        layout = self.parameter_layouts[parameter]
        mesh = layout.mesh
        for axis, name in enumerate(mesh.mesh_dim_names or ()):
            if name in ("fsdp_shard", "edp_shard", "fsdp", "dp", "dp_shard"):
                return mesh, axis
        raise ValueError("Graph checkpoint cannot resolve the parameter's FSDP shard axis")

    def prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return detached, unpadded DCP views carrying global chunk metadata."""
        return self._map(payload, restoring=False)

    def restore(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Restore physical padding before the existing model/optimizer loaders run."""
        return self._map(payload, restoring=True)

    def _owner(self, key: Any, current: str | None) -> str | None:
        """Resolve ordinary, flattened optimizer and main-weight FQN keys."""
        if isinstance(key, str):
            if key in self.parameters:
                return key
            if key.startswith("state."):
                candidate = key[len("state."):].rsplit(".", 1)[0]
                if candidate in self.parameters:
                    return candidate
        return current

    def _tensor_view(self, value: torch.Tensor, owner: str, path: tuple, restoring: bool) -> torch.Tensor:
        """Add a chunk to an unpadded view, or reconstruct physical padding on load."""
        parameter = self.parameters[owner]
        if value.ndim == 0 and parameter.ndim != 0:
            return value
        chunk = self.chunks[owner]
        expected = chunk.chunk.sizes if restoring else tuple(parameter.shape)
        if tuple(value.shape) != expected:
            raise ValueError(f"Graph checkpoint tensor {path} has shape {tuple(value.shape)}; expected {expected}")
        if restoring:
            if tuple(value.shape) == tuple(parameter.shape):
                return value
            result = value.new_zeros(parameter.shape)
            result[tuple(slice(0, size) for size in value.shape)].copy_(value)
            return result
        result = value.detach()[tuple(slice(0, size) for size in chunk.chunk.sizes)]
        setattr(result, CHUNK_INFO, chunk)
        return result

    def _map(self, payload: dict[str, Any], *, restoring: bool) -> dict[str, Any]:
        """Transform only tensor-owner payloads, preserving other training state."""
        def visit(value: Any, owner: str | None = None, path: tuple = ()) -> Any:
            """Copy the payload containers and visit their tensor leaves."""
            if isinstance(value, dict):
                result = copy(value)
                for key, item in value.items():
                    result[key] = visit(item, self._owner(key, owner), (*path, key))
                return result
            if isinstance(value, (list, tuple)):
                return type(value)(visit(item, owner, (*path, index)) for index, item in enumerate(value))
            if not isinstance(value, torch.Tensor):
                return value
            if owner is None:
                if path and path[0] == "optimizer" and value.ndim:
                    raise ValueError("Graph checkpoint requires FQN-keyed optimizer state")
                return value
            return self._tensor_view(value, owner, path, restoring)

        result = copy(payload)
        for key in ("model", "optimizer"):
            if key in payload:
                result[key] = visit(payload[key], path=(key,))
        return result

    @torch.no_grad()
    def full_model_state(self, *, keep: bool) -> dict[str, Any]:
        """Gather FSDP first, remove its padding, then gather the source TP/EP layout."""
        result = {}
        for name, value in self.compiler.model.state_dict(keep_vars=True).items():
            parameter = self.parameters.get(name)
            if isinstance(value, DTensor):
                value = value.full_tensor()
            elif parameter is not None:
                parameter_name = self.canonical_names[id(parameter)]
                value = value.detach()
                if parameter_name in self.compiler.sharded_parameter_names:
                    mesh, axis = self._shard_axis(parameter)
                    group = mesh.get_group(axis)
                    gathered = value.new_empty((value.shape[0] * dist.get_world_size(group), *value.shape[1:]))
                    dist.all_gather_into_tensor(gathered, value.contiguous(), group=group)
                    value = gathered[:self.parameter_shapes[parameter_name][0]]
                source = self.compiler.source_layout_for_parameter(parameter_name)
                if source is not None and source[1] is not None:
                    layout = Layout.from_device_mesh(source[1])
                    layout.set_placements(tuple(source[0]))
                    layout.placement_to_tensor_map(value.ndim)
                    value = DTensor.from_local_with_layout(value, layout).full_tensor()
            else:
                layout = getattr(value, "_sharding_spec", None)
                if isinstance(value, torch.Tensor) and layout is not None:
                    value = DTensor.from_local_with_layout(value.detach(), layout).full_tensor()
            if keep:
                result[name] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
        return result
