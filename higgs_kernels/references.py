import torch


def higgs_dequantize_2_256_torch(x, grid):
    assert grid.dtype in (torch.float16, torch.bfloat16)
    assert x.dtype == torch.uint8
    assert grid.shape == (256, 2)

    (out_dim,) = x.shape
    assert out_dim > 0

    out = grid[x.to(torch.int32)]

    assert out.shape == (out_dim, 2)
    assert out.dtype == grid.dtype

    return out


def higgs_quantize_2_256_torch(x, grid, grid_norms):
    assert x.dtype == grid.dtype == grid_norms.dtype
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert grid.shape == (256, 2), grid.shape
    assert grid_norms.shape == (256,), grid_norms.shape
    out_dim, in_dim = x.shape
    assert in_dim == 2
    assert out_dim > 0

    return torch.argmax(2 * x @ grid.T - grid_norms, dim=-1).to(torch.uint8)
