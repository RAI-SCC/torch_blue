import pytest
import torch
from torch import nn

from torch_blue.vi import convert_to_vimodule


@pytest.mark.parametrize(
    "module,n_args",
    [
        (nn.Linear(5, 6, bias=True), 1),
        (nn.Linear(5, 6, bias=False), 1),
        (nn.MultiheadAttention(5, 1), 3),
        (nn.Transformer(5, 1, 1, 1, 5), 2),
    ],
)
def test_convert_to_vimodule(module: nn.Module, n_args: int) -> None:
    """Test autoconversion to vi module."""
    sample = []
    for _ in range(n_args):
        sample.append(torch.randn(2, 3, 5))

    convert_to_vimodule(module)

    module(*sample)
