import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


# ge_scalar kernel
@libentry()
@triton.autotune(configs=[
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
], key=["N"])
@triton.jit
def ge_scalar_kernel(
    A_ptr, OUT_ptr, N,
    B,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask)
    # Convert to float32 for comparison (matching baseline behavior)
    a_f32 = a.to(tl.float32)
    b_f32 = B.to(tl.float32)
    result = a_f32 >= b_f32
    tl.store(OUT_ptr + offs, result, mask=mask)


def ge_scalar(A: torch.Tensor, B) -> torch.Tensor:
    logger.debug("GEMS_ILUVATAR GE_SCALAR")
    N = volume(A.shape)
    if N == 0:
        return torch.empty_like(A, dtype=torch.bool)

    A_flat = A.contiguous().view(-1)
    output = torch.empty_like(A_flat, dtype=torch.bool)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    ge_scalar_kernel[grid](
        A_flat, output, N, B
    )
    return output.view(A.shape)


# ge (tensor-tensor) kernel
@libentry()
@triton.autotune(configs=[
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
], key=["N"])
@triton.jit
def ge_kernel(
    A_ptr, B_ptr, OUT_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask)
    b = tl.load(B_ptr + offs, mask=mask)
    # Convert to float32 for comparison (matching baseline behavior)
    a_f32 = a.to(tl.float32)
    b_f32 = b.to(tl.float32)
    result = a_f32 >= b_f32
    tl.store(OUT_ptr + offs, result, mask=mask)


def ge(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ILUVATAR GE")
    N = volume(A.shape)
    if N == 0:
        return torch.empty_like(A, dtype=torch.bool)

    A_flat = A.contiguous().view(-1)
    B_flat = B.contiguous().view(-1)
    output = torch.empty_like(A_flat, dtype=torch.bool)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    ge_kernel[grid](
        A_flat, B_flat, output, N
    )
    return output.view(A.shape)
