import torch
from torch.nn import Module, Parameter

from torchbuq.vi.utils import init


def test_fixed(device: torch.device) -> None:
    """Test init.fixed_()."""
    param_shape = (5, 4)

    class ModuleDummy(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_mean = Parameter(torch.empty(param_shape, device=device))

    dummy = ModuleDummy()

    iter1 = dummy.parameters()
    other1 = torch.randn(param_shape, requires_grad=False, device=device)
    assert not torch.allclose(dummy.weight_mean, other1)

    init.fixed_(dummy.weight_mean, other1)
    weight1 = iter1.__next__().clone()
    assert torch.allclose(dummy.weight_mean, other1)
    assert torch.allclose(weight1, other1)
    assert weight1.requires_grad
    assert weight1.device == device
