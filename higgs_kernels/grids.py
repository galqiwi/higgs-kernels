import functools
import os

import safetensors.torch
import torch

PKG_PATH = os.path.dirname(os.path.realpath(__file__))


@functools.cache
def load_optimal_grid_2_256(device="cpu", dtype=torch.float16):
    return safetensors.torch.load_file(
        os.path.join(PKG_PATH, "grids.safetensors"), device=device
    )["2_256"].to(dtype)
