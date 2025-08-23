import pytest
import torch

from higgs_kernels.hadamard import hadamard_transform


def test_hadamard_matrix_4():
    reference = (
        torch.tensor([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1])
        .reshape(4, 4)
        .to(torch.int64)
    )
    torch.testing.assert_close(
        hadamard_transform(torch.eye(4, dtype=torch.int64)), reference
    )


def hadamard_transform_matmul(x):
    out_dim, in_dim = x.shape
    return x @ hadamard_transform(torch.eye(in_dim, dtype=x.dtype))


@pytest.mark.parametrize("in_dim", [1, 2, 4, 128])
@pytest.mark.parametrize("out_dim", [1, 2, 3])
def test_hadamard(in_dim, out_dim):
    x = torch.randn((out_dim, in_dim), dtype=torch.float32)
    torch.testing.assert_close(hadamard_transform(x), hadamard_transform_matmul(x))
