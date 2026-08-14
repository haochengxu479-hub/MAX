# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


# ==================== 第一阶段归约内核 ====================
@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=min_value)
    max_val = tl.max(inp_val)
    tl.store(mid + pid, max_val)


# ==================== 第二阶段归约内核 ====================
@libentry()
@triton.jit
def max_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid + offset, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


# ==================== 全局最大值（多级归约，支持 float16） ====================
def max(inp):
    logger.debug("GEMS MAX")
    inp = inp.contiguous()
    M = inp.numel()
    if M == 0:
        if inp.dtype.is_floating_point:
            min_val = torch.finfo(inp.dtype).min
        else:
            min_val = torch.iinfo(inp.dtype).min
        return torch.tensor(min_val, dtype=inp.dtype, device=inp.device)

    # ---------- 第一级归约 ----------
    block_size = min(1024, triton.next_power_of_2(M))
    mid_size = triton.cdiv(M, block_size)
    mid = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
    grid = (mid_size,)
    with torch_device_fn.device(inp.device):
        max_kernel_1[grid](inp, mid, M, block_size)

    # ---------- 多级归约：如果 mid_size > 1024，继续压缩 ----------
    while mid_size > 1024:
        new_block_size = min(1024, triton.next_power_of_2(mid_size))
        new_mid_size = triton.cdiv(mid_size, new_block_size)
        new_mid = torch.empty((new_mid_size,), dtype=inp.dtype, device=inp.device)
        grid_new = (new_mid_size,)
        max_kernel_1[grid_new](mid, new_mid, mid_size, new_block_size)
        mid = new_mid
        mid_size = new_mid_size

    # ---------- 最后一级归约（此时 mid_size <= 1024） ----------
    block_mid = triton.next_power_of_2(mid_size)
    out = torch.empty([], dtype=inp.dtype, device=inp.device)
    max_kernel_2[(1,)](mid, out, mid_size, block_mid)
    return out


# ==================== 维度最大值（带索引） ====================
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def max_dim_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IDX_TYPE: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)
    result_index = tl.zeros([BLOCK_M], dtype=IDX_TYPE)

    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        max_index = max_index.to(IDX_TYPE)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)

    mask1 = m_offset < M
    tl.store(out_value + m_offset, result_value, mask=mask1)
    tl.store(out_index + m_offset, result_index, mask=mask1)


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim

    # 压缩数据到 [M, N] 连续内存（保证合并访问）
    inp = dim_compress(inp, dim)
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N

    # 根据 N 动态选择索引类型（节省显存和寄存器）
    if N < (1 << 31):
        idx_dtype = torch.int32
        triton_idx_dtype = tl.int32
    else:
        idx_dtype = torch.int64
        triton_idx_dtype = tl.int64

    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=idx_dtype, device=inp.device)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    # 手动固定块大小，避免 libtuner 冷启动开销（确保性能稳定）
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M),)

    with torch_device_fn.device(inp.device):
        max_dim_kernel[grid](
            inp,
            out_value,
            out_index,
            M,
            N,
            IDX_TYPE=triton_idx_dtype,
        )

    Max_out = namedtuple("max", ["values", "indices"])
    return Max_out(values=out_value, indices=out_index)