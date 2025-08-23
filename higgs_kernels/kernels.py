import functools
import os

import torch
from torch.utils.cpp_extension import load

PKG_PATH = os.path.dirname(os.path.realpath(__file__))


@functools.cache
def get_dequantize_kernels():
    return load(
        name="higgs_dequant",
        sources=[
            os.path.join(PKG_PATH, "./csrc/portable/dequant.cpp"),
            os.path.join(PKG_PATH, "./csrc/portable/dequant.cu"),
        ],
        verbose=False,
    )


@functools.cache
def get_quantize_kernels():
    return load(
        name="higgs_quant",
        sources=[
            os.path.join(PKG_PATH, "./csrc/portable/quant.cpp"),
            os.path.join(PKG_PATH, "./csrc/portable/quant_f16.cu"),
            os.path.join(PKG_PATH, "./csrc/portable/quant_bf16.cu"),
        ],
        verbose=False,
    )


def higgs_dequantize_2_256_kernel(x, grid):
    x = x.contiguous()
    grid = grid.contiguous()

    assert grid.device == x.device
    assert "cuda" in str(x.device)

    assert grid.dtype in (torch.float16, torch.bfloat16)
    assert grid.shape == (256, 2)

    assert x.dtype == torch.uint8
    (out_dim,) = x.shape
    assert out_dim > 0

    out = torch.zeros((out_dim, 2), dtype=grid.dtype, device=grid.device)

    get_dequantize_kernels().higgs_dequantize_2_256_ptr_cuda_portable(
        x.data_ptr(), grid.data_ptr(), out.data_ptr(), out_dim
    )

    return out


def higgs_quantize_2_256_kernel(x, grid, grid_norms):
    assert x.dtype == grid.dtype == grid_norms.dtype
    assert x.device == grid.device == grid_norms.device
    assert "cuda" in str(x.device)

    assert x.dtype in (torch.bfloat16, torch.float16)

    assert grid.shape == (256, 2), grid.shape
    assert grid_norms.shape == (256,), grid_norms.shape

    out_dim, in_dim = x.shape
    assert in_dim == 2
    assert out_dim > 0

    out = torch.empty((out_dim,), dtype=torch.uint8, device=x.device)

    kernel = {
        torch.bfloat16: get_quantize_kernels().higgs_quantize_2_256_ptr_bf16_cuda_portable,
        torch.float16: get_quantize_kernels().higgs_quantize_2_256_ptr_f16_cuda_portable,
    }[x.dtype]

    kernel(
        int(x.data_ptr()),
        int(grid.data_ptr()),
        int(grid_norms.data_ptr()),
        int(out.data_ptr()),
        out_dim,
    )

    return out
