from math import log
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import init

from torch_bayesian.vi import _globals

from ..utils import init as vi_init
from .base import Prior

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


class BasicQuietPrior(Prior):
    """
    Prior assuming normal distributed mean and std proportional to it.

    This is an experimental prior that assumes independent normal distributed weights.
    However, instead of each having the same mean and standard deviation it assumes
    the means to be normal distributed amd each standard deviation to be proportional to
    the respective mean by a fixed factor.

    The distribution parameters are "mean" and "log_std".

    This prior requires the parameter mean to calculate :meth:`~log_prob`.

    Parameters
    ----------
    std_ratio: float, default: 1.0
        The fixed ratio between each pair of standard deviation and mean.
    mean_mean: float, default: 0.0
        The mean of the distribution of means.
    mean_std: float, default: 1.0
        The standard deviation of the distribution of means.
    eps: float, default: 1e-10
        Epsilon for numerical stability.
    """

    def __init__(
        self,
        std_ratio: float = 1.0,
        mean_mean: float = 0.0,
        mean_std: float = 1.0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.distribution_parameters = ("mean", "log_std")
        self._required_parameters = ("mean",)
        self._scaling_parameters = ("mean_mean", "mean_std", "eps")
        self._std_ratio = std_ratio
        self.mean_mean = mean_mean
        self.mean_std = mean_std
        self.eps = eps

    def log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
        """
        Compute the log probability of the sample based on the prior.

        This calculates the Gaussian log probability of a sample using the current best
        estimate for its mean and adds a factor to account for the distribution of
        means.

        All Tensors have the same shape.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_bayesian.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log probability for.
        mean: Tensor
            The current best estimate for the mean of ech value.

        Returns
        -------
        Tensor
            The log probability of the sample under the prior.

        """
        variance = (self._std_ratio * mean) ** 2 + self.eps
        data_fitting = (sample - mean) ** 2 / variance
        mean_decay = (mean - self.mean_mean) ** 2 / (self.mean_std**2)
        normalization = variance.log() + 2 * log(self.mean_std)
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + 2 * log(2 * torch.pi)
        return -0.5 * (data_fitting + mean_decay + normalization)

    def reset_parameters(self, module: "VIModule", variable: str) -> None:
        """
        Reset the parameters of the module to prior mean and standard deviation.

        This samples the means from the specified mean distribution and initializes the
        corresponding log standard deviations to according the ratio given by
        :attr:`~std_ratio`.

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
        init._no_grad_normal_(getattr(module, mean_name), self.mean_mean, self.mean_std)
        log_std_name = module.variational_parameter_name(variable, "log_std")
        log_std = torch.log(self._std_ratio * getattr(module, mean_name).abs())
        vi_init.fixed_(getattr(module, log_std_name), log_std)
