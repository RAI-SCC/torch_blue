from math import log

import torch
from pytest import mark
from torch.nn import Module, Parameter

from torch_bayesian.vi.distributions import BasicQuietPrior
from torch_bayesian.vi.utils import use_norm_constants


@mark.parametrize("norm_constants", [True, False])
def test_prior_log_prob(norm_constants: bool, device: torch.device) -> None:
    """Test BasicQuietPrior.prior_log_prob()."""
    use_norm_constants(norm_constants)

    std_ratio1 = 0.5
    mean_mean1 = 0.1
    mean_std1 = 0.4
    eps1 = 1e-6
    prior1 = BasicQuietPrior(std_ratio1, mean_mean1, mean_std1, eps1)
    assert prior1.distribution_parameters == ("mean", "log_std")
    assert prior1._required_parameters == ("mean",)
    assert prior1._scaling_parameters == ("mean_mean", "mean_std", "eps")
    assert prior1._std_ratio == std_ratio1
    assert prior1.mean_mean == mean_mean1
    assert prior1.mean_std == mean_std1
    assert prior1.eps == eps1

    std_ratio2 = 1.0
    mean_mean2 = 0.0
    mean_std2 = 1.0
    eps2 = 1e-10
    prior2 = BasicQuietPrior()
    assert prior2._std_ratio == std_ratio2
    assert prior2.mean_mean == mean_mean2
    assert prior2.mean_std == mean_std2
    assert prior2.eps == eps2

    mean = torch.arange(0.1, 1.1, 0.1, device=device)
    sample1 = torch.zeros_like(mean, device=device)
    sample2 = mean.clone()
    variance = (std_ratio2 * mean) ** 2 + eps2

    ref1 = (
        -0.5 * mean**2 / variance
        - 0.5 * mean**2
        - 0.5 * variance.log()
        - log(2 * torch.pi)
    )
    if not norm_constants:
        norm_const = torch.full_like(mean, 2 * torch.pi, device=device).log()
        ref1 += norm_const
    log_prob1 = prior2.prior_log_prob(sample1, mean)
    assert torch.allclose(log_prob1, ref1)
    assert log_prob1.device == device

    ref2 = -0.5 * mean**2 - 0.5 * variance.log() - log(2 * torch.pi)
    if not norm_constants:
        norm_const = torch.full_like(mean, 2 * torch.pi, device=device).log()
        ref2 += norm_const
    log_prob2 = prior2.prior_log_prob(sample2, mean)
    assert torch.allclose(log_prob2, ref2)
    assert log_prob2.device == device


def test_reset_variational_parameters(device: torch.device) -> None:
    """Test BasicQuietPrior.reset_parameters_to_prior()."""
    param_shape = (500, 400)

    class ModuleDummy(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_mean = Parameter(torch.empty(param_shape))
            self.weight_log_std = Parameter(torch.empty(param_shape))

        @staticmethod
        def variational_parameter_name(variable: str, parameter: str) -> str:
            return f"{variable}_{parameter}"

    prior = BasicQuietPrior(0.5, -1.0, 5.0)
    dummy = ModuleDummy().to(device=device)
    mean0 = dummy.weight_mean.clone()

    iter1 = dummy.parameters()
    prior.reset_parameters_to_prior(dummy, "weight")

    mean = iter1.__next__().clone()
    log_std = iter1.__next__().clone()

    assert not torch.allclose(mean0, mean)
    assert torch.allclose(mean.mean(), torch.tensor(-1.0, device=device), atol=1e-1)
    assert torch.allclose(mean.std(), torch.tensor(5.0, device=device), rtol=1e-1)
    assert (log_std == (mean.abs() / 2).log()).all()
    assert mean.device == device
    assert log_std.device == device
