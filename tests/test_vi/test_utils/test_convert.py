from copy import deepcopy

import pytest
import torch
from torch import nn

import torch_blue
from torch_blue.vi import convert_to_vimodule
from torch_blue.vi.distributions import MeanFieldNormal, NonBayesian, StudentT
from torch_blue.vi.utils import convert


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

    module1 = deepcopy(module)
    convert_to_vimodule(module1)
    ref_class = getattr(torch_blue.vi, module1.__class__.__name__)
    assert ref_class == type(module1)
    module1(*sample)

    convert.ban_torch_convert(module.__class__.__name__)
    module2 = deepcopy(module)
    convert_to_vimodule(module2)
    assert ref_class != type(module2)
    module2(*sample)

    module3 = deepcopy(module)
    convert_to_vimodule(module3)
    assert type(module2) == type(module3)
    module3(*sample)

    convert.ban_reuse(module.__class__.__name__)
    module4 = deepcopy(module)
    convert_to_vimodule(module4)
    assert type(module4) is not type(module3)
    module4(*sample)
    convert.ban_reuse(module.__class__.__name__, False)
    convert.ban_torch_convert(module.__class__.__name__, False)

    convert.ban_convert(type(module))
    module9 = deepcopy(module)
    with pytest.raises(
        AttributeError,
        match=f"'{module.__class__.__name__}' object has no attribute '_set_sampling_responsibility'",
    ):
        convert_to_vimodule(module9)
    convert.ban_convert(type(module), False)


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


@pytest.mark.parametrize("ban_norms", [True, False])
def test_blacklist(ban_norms: bool) -> None:
    """Test basic blacklist and norm setting."""
    assert torch.nn.ReLU in convert._blacklist
    assert torch.nn.LayerNorm in convert._torch_norms

    convert.convert_norms(not ban_norms)
    if ban_norms:
        assert convert._blacklist & convert._torch_norms == convert._torch_norms
    else:
        assert convert._blacklist & convert._torch_norms == set()

    convert.convert_norms(True)
    assert convert._blacklist & convert._torch_norms == set()


@pytest.mark.parametrize("ban_norms", [True, False])
def test_ban_convert(ban_norms: bool) -> None:
    """Test adding and removing modules from blacklist."""
    convert.convert_norms(not ban_norms)

    module = torch.nn.Linear
    assert module not in convert._blacklist
    convert.ban_convert(module)
    assert module in convert._blacklist
    convert.ban_convert(module, False)
    assert module not in convert._blacklist

    module_list = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    for module in module_list:
        assert module not in convert._blacklist
    convert.ban_convert(module_list)
    for module in module_list:
        assert module in convert._blacklist
    convert.ban_convert(module_list, False)
    for module in module_list:
        assert module not in convert._blacklist


@pytest.mark.parametrize("mode", ["torch", "reuse"])
def test_reuse_bans(mode: str) -> None:
    """Test adding and removing modules from reuse lists."""
    if mode == "torch":
        lst = convert._torch_blacklist
        method = convert.ban_torch_convert
    elif mode == "reuse":
        lst = convert._reuse_blacklist
        method = convert.ban_reuse

    module = "Linear"
    assert module not in lst
    method(module)
    assert module in lst
    method(module, False)
    assert module not in lst

    module_list = ["Conv1d", "Conv2d", "Conv3d"]
    for module in module_list:
        assert module not in lst
    method(module_list)
    for module in module_list:
        assert module in lst
    method(module_list, False)
    for module in module_list:
        assert module not in lst


@pytest.mark.parametrize("keep_weights", [True, False])
def test_dtype_copying(keep_weights: bool, device: torch.device) -> None:
    """
    Test copying of inconsistent dtypes during autoconversion.

    This can also be considered as a stand-in for testing copying of inconsistent
    devices, which follows the same pattern but is quite hard to test due to hardware
    requirements.
    """
    # Set up split dtype/device layer and verify
    model = nn.Linear(3, 5, bias=True, device=device, dtype=torch.float16)
    model.bias.data = model.bias.to(device="cpu", dtype=torch.float32)
    assert model.weight.device == device
    assert model.bias.device == torch.device("cpu")
    assert model.weight.dtype == torch.float16
    assert model.bias.dtype == torch.float32

    # convert and verify
    convert_to_vimodule(
        model,
        variational_distribution=(MeanFieldNormal(), StudentT()),
        keep_weights=keep_weights,
    )
    assert model.weight.device == device
    assert model.bias.device == torch.device("cpu")
    assert model.weight.dtype == torch.float16
    assert model.bias.dtype == torch.float32
