import torch

from higgs_kernels.grids import load_optimal_grid_2_256
from higgs_kernels.linear import (
    higgs_dequantize_linear_2_256,
    higgs_quantize_linear_2_256,
)


def higgs_quantize_dequantize(weight, grid, group_size=512):
    quantized, scales = higgs_quantize_linear_2_256(weight, grid, group_size)
    return higgs_dequantize_linear_2_256(quantized, scales, grid, group_size)


def test_quantize_linear():
    weight = torch.randn((4096, 1024), dtype=torch.float16, device="cuda")
    grid = load_optimal_grid_2_256(dtype=torch.float16, device="cuda")

    new_weight = higgs_quantize_dequantize(weight, grid)
    assert torch.linalg.norm(new_weight - weight) / torch.linalg.norm(weight) < 0.15
