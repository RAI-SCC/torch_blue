from math import exp, log
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import init

from torch_bayesian.vi import _globals

from .base import Prior

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


class MeanFieldNormalPrior(Prior):
    """
    Prior assuming uncorrelated, normal distributed parameters.

    This prior assumes weights that are independently and identically normal distributed.

    When used with :class:`~torch_bayesian.vi.KullbackLeiblerLoss` or
    :class:`~torch_bayesian.vi.AnalyticalKullbackLeiblerLoss` and a
    :class:`~torch_bayesian.vi.variational_distributions.MeanFieldNormalVarDist`
    the prior matching term becomes equivalent to an L2-weight decay term.

    The distribution parameters are "mean" and "log_std".

    Parameters
    ----------
    mean: float, default: 0.0
        The mean of the normal distribution before potential rescaling.
    std: float, default: 1.0
        The standard deviation of the normal distribution before potential rescaling.
        This is converted to a log std internally.
    eps: float, default: 1e-10
        Epsilon for numerical stability.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-10) -> None:
        super().__init__()
        self.distribution_parameters = ("mean", "log_std")
        self.mean = mean
        self.log_std = log(std)
        self.eps = eps

    @property
    def std(self) -> float:
        """Prior standard deviation."""
        return exp(self.log_std)

    def log_prob(self, sample: Tensor) -> Tensor:
        """
        Compute the Gaussian log probability of a sample using the prior parameters.

        All Tensors have the same shape.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_bayesian.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log prbability for.

        Returns
        -------
        Tensor
            The log probability of the sample under the prior.
        """
        variance = self.std**2 + self.eps
        data_fitting = (sample - self.mean) ** 2 / variance
        normalization = 2 * self.log_std
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    def reset_parameters(self, module: "VIModule", variable: str) -> None:
        """
        Reset the parameters of a module to prior mean and standard deviation.

        Parameters
        ----------
        module: VIModule
            The module containing the parameters to reset.
        variable: str
            The name of the random variable to reset as given by
            :attr:`variational_parameters` of the associated
            :class:`~torch_bayesian.vi.variational_distributions.VariationalDistribution`.

        Returns
        -------
        None
        """
        mean_name = module.variational_parameter_name(variable, "mean")
        init.constant_(getattr(module, mean_name), self.mean)
        log_std_name = module.variational_parameter_name(variable, "log_std")
        init.constant_(getattr(module, log_std_name), self.log_std)
