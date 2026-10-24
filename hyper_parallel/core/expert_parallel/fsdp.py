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
"""Expert parameter placement shared by eager and graph FSDP."""

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard


def expert_shard_placement(parameter: torch.Tensor, *, shard_degree: int) -> Shard:
    """Choose an evenly divisible expert or inner parameter dimension.

    Args:
        parameter: EP-local expert parameter before FSDP sharding.
        shard_degree: Size of the expert data-parallel shard group.

    Returns:
        A shard placement, preferring the expert dimension over inner dimensions.

    Raises:
        ValueError: If the degree is invalid or no dimension can be evenly sharded.
    """
    if shard_degree < 1:
        raise ValueError(f"Expert FSDP shard degree must be positive, got {shard_degree}")
    if isinstance(parameter, DTensor):
        parameter = parameter.to_local()
    for dim, size in enumerate(parameter.shape):
        if size > 0 and size % shard_degree == 0:
            return Shard(dim)
    raise ValueError(
        f"Expert parameter with shape {tuple(parameter.shape)} has no divisible dimension "
        f"for edp_shard degree {shard_degree}"
    )
