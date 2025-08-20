from math import log
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import StudentT

from torch_bayesian.vi import _globals

from .base import VariationalDistribution


class StudentTVarDist(VariationalDistribution):
    """
    Variational distribution of independent Student's t-distributions.

    Defines a variational Student's t-distribution with initial mean zero. Learnable
    parameters are mean and log_scale; degrees_of_freedom can be provided but is fixed.

    Parameters
    ----------
    initial_scale: float, default: 1.0
        initial scale each independent distribution.
    degrees_of_freedom: float, default: 4.0
        degrees of freedom each independent distribution.
    """

    def __init__(
        self,
        initial_scale: float = 1.0,
        degrees_of_freedom: float = 4.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.degrees_of_freedom = torch.tensor(degrees_of_freedom, device=device)
        self.variational_parameters = ("mean", "log_scale")
        self._default_variational_parameters = (0.0, log(initial_scale))

    def sample(self, mean: Tensor, log_scale: Tensor) -> Tensor:
        """
        Draw sample from Student's t-distribution.

        Draw samples from a Student's t-distribution with the given mean and log_scale.
        mean and log_scale must be broadcastable to the same shape.

        Parameters
        ----------
        mean: Tensor
            Sample mean.
        log_scale: Tensor
            Sample distribution log scale.

        Returns
        -------
        sample: Tensor
            Sample tensor of the same shape as ``mean`` drawn from Student's t-distribution.
        """
        scale = torch.exp(log_scale)
        return self._student_t_sample(mean, scale)

    def log_prob(self, sample: Tensor, mean: Tensor, log_scale: Tensor) -> Tensor:
        """
        Compute the log probability of a sample.

        Computes the log probability of sample in a Student's t-distribution with the
        given mean and log_scale.

        Parameters
        ----------
        sample: Tensor
            Sample tensor.
        mean: Tensor
            Distribution mean.
        log_scale: Tensor
            Distribution log scale.

        Returns
        -------
        log_prob: Tensor
            Tensor with the same shape as ``sample`` containing the log probability of the sample given ``mean`` and ``log_scale``.
        """
        self.degrees_of_freedom = self.degrees_of_freedom.to(device=sample.device)
        scale = torch.exp(log_scale)
        data_fitting = (
            (self.degrees_of_freedom + 1.0)
            * 0.5
            * torch.log(1.0 + ((sample - mean) / scale) ** 2 / self.degrees_of_freedom)
        )
        normalization = log_scale
        if _globals._USE_NORM_CONSTANTS:
            normalization = (
                normalization
                + 0.5 * log(torch.pi * self.degrees_of_freedom)
                + torch.lgamma(self.degrees_of_freedom / 2.0)
                - torch.lgamma((self.degrees_of_freedom + 1.0) / 2.0)
            )
        return -(data_fitting + normalization)

    def _student_t_sample(self, mean: Tensor, scale: Tensor) -> Tensor:
        """
        Draw sample from Student's t-distribution.

        Draw samples from a Student's t-distribution with the given mean and scale.
        mean and log_scale must be broadcastable to the same shape.

        Parameters
        ----------
        mean: Tensor
            Sample mean.
        scale: Tensor
            Sample distribution scale.

        Returns
        -------
        sample: Tensor
            Sample tensor of the same shape as ``mean`` drawn from Student's t-distribution.
        """
        self.degrees_of_freedom = self.degrees_of_freedom.to(device=mean.device)
        base_sample = StudentT(self.degrees_of_freedom).sample(sample_shape=mean.shape)
        sample = scale * base_sample + mean
        return sample
