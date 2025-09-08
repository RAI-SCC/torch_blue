from math import log

import torch
from torch import Tensor

from torch_bayesian.vi import _globals

from .base import VariationalDistribution


class MeanFieldNormalVarDist(VariationalDistribution):
    """
    A variational distribution with uncorrelated, normal distributions.

    Following the mean field approximation, this variational distribution prescribes
    that each weight is independently normal distributed with its own mean and standard
    deviation. This is often the default assumption for BNNs.

    This distribution's variational parameters are `mean` and `log_std`.
    """

    def __init__(self, initial_std: float = 1.0) -> None:
        super().__init__()
        self.variational_parameters = ("mean", "log_std")
        self._default_variational_parameters = (0.0, log(initial_std))

    def sample(self, mean: Tensor, log_std: Tensor) -> Tensor:
        """
        Sample from a Gaussian distribution.

        Parameters
        ----------
        mean: Tensor
            The mean for each sample as Tensor.
        log_std: Tensor
            The log standard deviation for each sample as Tensor. Must have the same
            shape as `mean`.

        Returns
        -------
        Tensor
            The sampled Tensor of teh same shape as `mean`.
        """
        std = torch.exp(log_std)
        return self._normal_sample(mean, std)

    def log_prob(self, sample: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        """
        Compute the log probability of `sample` based on a normal distribution.

        Calculates the log probability of `sample` based on the provided mean and log
        standard deviation. All Tensors must have the same shape as `sample`.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_bayesian.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            The weight configuration to calculate the log probability for.
        mean: Tensor
            The means of the reference distribution.
        log_std: Tensor
            The log standard deviations of the reference distribution.

        Returns
        -------
        Tensor
            The log probability of `sample` based on the provided mean and log_std.
        """
        variance = torch.exp(log_std) ** 2
        data_fitting = (sample - mean) ** 2 / variance
        normalization = 2 * log_std
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    @staticmethod
    def _normal_sample(mean: Tensor, std: Tensor) -> Tensor:
        base_sample = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        sample = std * base_sample + mean
        return sample
