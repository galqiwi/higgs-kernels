import torch

from higgs_kernels.hadamard import hadamard_transform
from higgs_kernels.references import (
    higgs_dequantize_2_256_torch,
    higgs_quantize_2_256_torch,
)


def higgs_quantize_linear_2_256(weight, grid, group_size):
    assert weight.device == grid.device

    out_dim, in_dim = weight.shape
    assert in_dim % group_size == 0
    assert group_size % 2 == 0

    assert grid.shape == (256, 2)

    assert weight.dtype == grid.dtype
    assert weight.dtype in (torch.float16, torch.bfloat16)

    grid_norms = (torch.linalg.norm(grid, dim=-1) ** 2).to(grid.dtype)

    groups_per_input = in_dim // group_size

    weight = weight.reshape(out_dim, groups_per_input, group_size)
    scales = torch.linalg.norm(weight, dim=-1) / group_size**0.5
    assert scales.shape == (out_dim, groups_per_input)

    weight = weight / scales[:, :, None]

    weight = hadamard_transform(weight) / group_size**0.5

    weight = weight.reshape(-1, 2)

    quantized = higgs_quantize_2_256_torch(weight, grid, grid_norms)
    assert quantized.shape == (out_dim * in_dim // 2,)
    assert quantized.dtype == torch.uint8

    return quantized, scales


def higgs_dequantize_linear_2_256(quantized, scales, grid, group_size):
    out_dim, groups_per_input = scales.shape
    in_dim = groups_per_input * group_size
    assert quantized.shape == (out_dim * in_dim // 2,)

    assert quantized.dtype == torch.uint8
    assert scales.dtype in (torch.float16, torch.bfloat16)
    assert grid.dtype == scales.dtype
    assert scales.device == quantized.device == grid.device

    output = higgs_dequantize_2_256_torch(quantized, grid)
    assert output.shape == (out_dim * in_dim // 2, 2)

    output = output.reshape(out_dim, groups_per_input, group_size)

    output = hadamard_transform(output) / group_size**0.5

    output = output * scales[:, :, None]

    output = output.reshape(out_dim, in_dim)

    return output


def higgs_matmul_linear_2_256(input, quantized, scales, grid, group_size):
    dequantized = higgs_dequantize_linear_2_256(quantized, scales, grid, group_size)
    return torch.nn.functional.linear(input, dequantized)
