import pytest
import torch
from torch import nn

from torch_blue.vi import convert_to_vimodule
from torch_blue.vi.distributions import NonBayesian


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


@pytest.mark.parametrize(
    "module,n_args",
    [
        (nn.Linear(5, 6, bias=True), 1),
        (nn.Linear(5, 6, bias=False), 1),
        (nn.MultiheadAttention(5, 1, dropout=0.0), 3),
        (nn.Transformer(5, 1, 1, 1, 5, dropout=0.0), 2),
    ],
)
def test_keep_weights(module: nn.Module, n_args: int) -> None:
    """Test weight keeping during autoconversion to vi module."""
    sample = []
    for _ in range(n_args):
        sample.append(torch.randn(2, 3, 5))

    ref = module(*sample)
    convert_to_vimodule(
        module, variational_distribution=NonBayesian(), keep_weights=True
    )

    out = module(*sample)
    if isinstance(ref, tuple):
        assert torch.allclose(ref[0], out[0])
        assert torch.allclose(ref[1], out[1])
    else:
        assert torch.allclose(ref, out)
