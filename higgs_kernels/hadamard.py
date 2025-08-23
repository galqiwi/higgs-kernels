import torch


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@torch.no_grad()
def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    In-place FWHT along the *last* dimension of `x`.
    Assumes size of last dim is a power of two.
    """
    x = x.detach().clone().contiguous()

    n = x.shape[-1]
    h = 1

    assert is_pow2(n)

    while h < n:
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]

        tmp = a.clone()
        a.copy_(tmp + b)
        b.copy_(tmp - b)

        x = x.view(*x.shape[:-3], n)
        h <<= 1

    return x
