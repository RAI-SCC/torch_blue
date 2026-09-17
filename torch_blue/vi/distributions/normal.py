from math import log
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import init

from torch_blue.vi import _globals

from .base import PredictiveDistribution, Prior, VariationalDistribution

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


class MeanFieldNormal(Prior, VariationalDistribution, PredictiveDistribution):
    """
    Distribution assuming uncorrelated, normal distributed values.

    This distribution is implemented as prior, variational distribution, and predictive
    distribution. Its distribution parameters are "mean" and "log_std".

    As prior it becomes equivalent to an L2-weight decay term int the Kullback-Leibler
    loss.

    As variational distribution it is often the default assumption.

    As predictive distribution makes the Kullback-Leibler loss similar to MSE loss.

    Parameters
    ----------
    mean: float, default: 0.0
        The mean of the normal distribution before potential rescaling. Ignored if used
        as predictive distribution.
    std: float, default: 1.0
        The standard deviation of the normal distribution before potential rescaling.
        This is converted to a log std internally. Ignored if used as predictive
        distribution.
    eps: float, default: 1e-10
        Epsilon for numerical stability.
    """

    distribution_parameters = ("mean", "log_std")

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-10) -> None:
        super().__init__()
        self.mean = torch.tensor(mean)
        self.log_std = torch.tensor(std).log()
        self.eps = eps

    @property
    def _default_variational_parameters(self) -> tuple[Tensor, Tensor]:
        return self.mean, self.log_std

    @property
    def std(self) -> Tensor:
        """Standard deviation of the distribution."""
        return self.log_std.exp()

    def log_prob(self, sample: Tensor, parameters: tuple[Tensor, Tensor]) -> Tensor:
        """
        Compute the log probability of `sample` based on a normal distribution.

        Calculates the log probability of `sample` based on the provided mean and log
        standard deviation. All Tensors must have the same shape as `sample`.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            The weight configuration to calculate the log probability for.
        parameters: tuple[Tensor, Tensor]
            The mean and the standard deviation of the distribution.

        Returns
        -------
        Tensor
            The log probability of `sample` based on the provided mean and log
            standard deviation.
        """
        mean, log_std = parameters
        variance = torch.exp(log_std) ** 2 + self.eps
        data_fitting = (sample - mean) ** 2 / variance
        normalization = variance.log()
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    def sample(self, parameters: tuple[Tensor, Tensor]) -> Tensor:
        """
        Sample from a Gaussian distribution.

        Parameters
        ----------
        parameters: tuple[Tensor, Tensor]
            The mean and the standard deviation of the distribution.

        Returns
        -------
        Tensor
            The sampled Tensor of the same shape as as the two input Tensors.
        """
        mean, log_std = parameters
        std = torch.exp(log_std)
        return self._normal_sample(mean, std)

    @staticmethod
    def _normal_sample(mean: Tensor, std: Tensor) -> Tensor:
        base_sample = torch.randn_like(mean)
        sample = std * base_sample + mean
        return sample

    def reset_parameters_to_prior(self, module: "VIModule", variable: str) -> None:
        """
        Reset the parameters of a module to prior mean and standard deviation.

        Parameters
        ----------
        module: VIModule
            The module containing the parameters to reset.
        variable: str
            The name of the random variable to reset as given by
            :attr:`distribution_parameters` of the associated
            :class:`~torch_blue.vi.distributions.Prior`.

        Returns
        -------
        None
        """
        mean_name = module.variational_parameter_name(variable, "mean")
        init.constant_(getattr(module, mean_name), self.mean.item())
        log_std_name = module.variational_parameter_name(variable, "log_std")
        init.constant_(getattr(module, log_std_name), self.log_std.item())

    def predictive_parameters_from_samples(
        self, samples: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""
        Calculate predictive mean and standard deviation of samples.

        Parameters
        ----------
        samples: Tensor
            The model output as Tensor of shape (S, \*), where S is the number of
            samples.

        Returns
        -------
        Tensor
            The predictive mean as Tensor of shape (\*), i.e., the average along the
            sample dimension.
        Tensor
            The predictive standard deviation as Tensor of shape (\*), i.e., the
            standard deviation along the sample dimension.
        """
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std

    def log_prob_from_samples(self, reference: Tensor, samples: Tensor) -> Tensor:
        r"""
        Calculate the log probability for reference given a set of samples.

        Since :meth`~predictive_parameters_from_samples` returns an std instead of a
        log_std it needs to be converted before being passed to :meth`~log_prob`.

        Parameters
        ----------
        reference : Tensor
            Expected prediction as Tensor of shape (\*)
        samples : Tensor
            Model prediction as Tensor of shape (S, \*), where S is the number of samples.

        Returns
        -------
        Tensor
            The log probability of the reference under the predicted distribution.
            Shape: (1,).
        """
        mean, std = self.predictive_parameters_from_samples(samples)
        log_prob = self.log_prob(reference, (mean, std.log()))
        return log_prob
