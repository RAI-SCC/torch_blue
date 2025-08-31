from math import log

import torch
from pytest import mark
from torch.distributions import Normal
from torch.nn import Module, Parameter

from torch_bayesian.vi.priors import MeanFieldNormalPrior
from torch_bayesian.vi.utils import use_norm_constants


@mark.parametrize("norm_constants", [True, False])
def test_log_prob(norm_constants: bool, device: torch.device) -> None:
    """Test MeanFieldNormalPrior.log_prob."""
    use_norm_constants(norm_constants)

    mean = 0.0
    std = 1.0
    prior = MeanFieldNormalPrior()
    assert prior.std == 1.0
    ref_dist = Normal(mean, std)
    shape1 = (3, 4)
    sample = ref_dist.sample(shape1).to(device=device)
    ref1 = ref_dist.log_prob(sample).to(device=device)
    if not norm_constants:
        norm_const = torch.full(shape1, 2 * torch.pi, device=device).log() / 2
        ref1 += norm_const
    log_prob1 = prior.log_prob(sample)
    assert torch.allclose(ref1, log_prob1, atol=1e-7)

    mean = 0.7
    std = 0.3
    eps = 1e-3
    prior = MeanFieldNormalPrior(mean, std, eps=eps)
    assert prior.std == std
    ref_dist = Normal(mean, std)
    shape2 = (6,)
    sample = ref_dist.sample(shape2).to(device=device)
    ref2 = -0.5 * (2 * log(std) + (sample - mean) ** 2 / (std**2 + eps))
    if norm_constants:
        norm_const = torch.full(shape2, 2 * torch.pi, device=device).log() / 2
        ref2 -= norm_const
    log_prob2 = prior.log_prob(sample)
    assert torch.allclose(ref2, log_prob2, atol=1e-7)


def test_normal_reset_variational_parameters(device: torch.device) -> None:
    """Test MeanFieldNormalPrior.reset_variational_parameters()."""
    param_shape = (5, 4)

    class ModuleDummy(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_mean = Parameter(torch.empty(param_shape))
            self.weight_log_std = Parameter(torch.empty(param_shape))

        @staticmethod
        def variational_parameter_name(variable: str, parameter: str) -> str:
            return f"{variable}_{parameter}"

    prior = MeanFieldNormalPrior(3.0, 2.0)
    dummy = ModuleDummy().to(device=device)

    iter1 = dummy.parameters()
    prior.reset_variational_parameters(dummy, "weight")

    mean = iter1.__next__().clone()
    log_std = iter1.__next__().clone()

    assert (mean == 3.0).all()
    assert (log_std == log(2.0)).all()
    assert mean.device == device
    assert log_std.device == device
