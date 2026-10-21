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

"""Functional fixed-split collectives used only by explicit graph EP."""

from typing import Sequence

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as fc


def ep_all_to_all(
    tokens: torch.Tensor,
    send_counts: Sequence[int],
    recv_counts: Sequence[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Exchange EP token chunks with statically known split sizes.

    Splits are source-rank ordered Python integers, fixed for every replay.
    The caller must derive a matching plan on all ranks.

    Args:
        tokens: Local token tensor ordered by destination rank.
        send_counts: Number of local rows sent to each rank.
        recv_counts: Number of rows received from each rank.
        group: EP process group.

    Returns:
        Received tokens ordered by source rank, with autograd support.

    Raises:
        RuntimeError: If distributed communication is not initialized.
        ValueError: If the token shape or static split sizes are invalid.

    Note:
        Changing split sizes after graph capture requires recompilation.
    """
    if tokens.ndim < 1:
        raise ValueError("EP tokens must have a leading token dimension")
    if not dist.is_initialized():
        raise RuntimeError("EP all-to-all requires an initialized process group")

    group_size = dist.get_world_size(group)
    for name, counts in (("send_counts", send_counts), ("recv_counts", recv_counts)):
        if len(counts) != group_size or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in counts
        ):
            raise ValueError(
                f"{name} must contain {group_size} non-negative Python integers"
            )
    if tokens.shape[0] != sum(send_counts):
        raise ValueError(
            f"EP token rows ({tokens.shape[0]}) must equal "
            f"sum(send_counts) ({sum(send_counts)})"
        )

    return fc.all_to_all_single_autograd(
        tokens.contiguous(), list(recv_counts), list(send_counts), group
    )
