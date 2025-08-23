import pytest
import torch

from higgs_kernels import higgs_dequantize_2_256_kernel, higgs_quantize_2_256_kernel
from higgs_kernels.references import (
    higgs_dequantize_2_256_torch,
    higgs_quantize_2_256_torch,
)


@pytest.mark.parametrize("N", [1, 2, 3, 4, 8, 500, 512, 1023, 1024, 1025, 1024**2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequantize(N, dtype):
    grid = torch.randn((256, 2), device="cuda", dtype=dtype)

    quantized = torch.randint(0, 256, (N,), dtype=torch.uint8, device="cuda")

    torch.testing.assert_close(
        higgs_dequantize_2_256_torch(quantized, grid),
        higgs_dequantize_2_256_kernel(quantized, grid),
    )


@pytest.mark.parametrize("N", [1, 2, 3, 4, 8, 500, 512, 1023, 1024, 1025, 1024**2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize(N, dtype):
    grid = torch.randn((256, 2), device="cuda", dtype=dtype)
    grid_norms = torch.linalg.norm(grid, dim=1) ** 2

    weight = torch.randn((N, 2), device="cuda", dtype=dtype)

    torch.testing.assert_close(
        higgs_quantize_2_256_torch(weight, grid, grid_norms),
        higgs_quantize_2_256_kernel(weight, grid, grid_norms),
    )
