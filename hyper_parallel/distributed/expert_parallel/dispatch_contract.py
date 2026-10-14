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
"""Expert-major rows shared by eager and captured expert execution."""

from dataclasses import dataclass
from typing import Union

import torch


@dataclass(frozen=True)
class ExpertBatch:
    """Rows grouped by local expert with exclusive end offsets.

    ``offsets`` contains monotonically increasing exclusive end positions,
    one per local expert; the last value equals ``tokens.shape[0]``. Eager
    routing uses a one-dimensional tensor with dynamic values. Graph routing
    uses a tuple of Python shape values, so tracing never reads a tensor on
    the host. Graph batches may include inactive padded rows; offsets express
    capacity, not token validity.

    Args:
        tokens: Expert-major rows with shape ``[rows, hidden]``.
        offsets: Exclusive end positions as a dynamic one-dimensional tensor
            or a static tuple of Python integers.
    """

    tokens: torch.Tensor
    offsets: Union[torch.Tensor, tuple[int, ...]]
