from math import exp, log
from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor
from torch.nn import init

from torch_blue.vi import _globals

from .base import Distribution

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


class MeanFieldNormal(Distribution):
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
        Epsilon for numerical stability. Only relevant if used as prior.
    """

    is_prior: bool = True
    is_variational_distribution: bool = True
    is_predictive_distribution: bool = True

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-10) -> None:
        super().__init__()
        self.distribution_parameters = ("mean", "log_std")
        self.mean = mean
        self.log_std = log(std)
        self.eps = eps
        self._default_variational_parameters = (0.0, log(std))

    @property
    def std(self) -> float:
        """Prior standard deviation."""
        return exp(self.log_std)

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

    @staticmethod
    def variational_log_prob(sample: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
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
        base_sample = torch.randn_like(mean)
        sample = std * base_sample + mean
        return sample

    def prior_log_prob(self, sample: Tensor) -> Tensor:
        """
        Compute the Gaussian log probability of a sample using the prior parameters.

        All Tensors have the same shape.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log probability for.

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

    def reset_parameters_to_prior(self, module: "VIModule", variable: str) -> None:
        """
        Reset the parameters of a module to prior mean and standard deviation.

        Parameters
        ----------
        module: VIModule
            The module containing the parameters to reset.
        variable: str
            The name of the random variable to reset as given by
            :attr:`variational_parameters` of the associated
            :class:`~torch_blue.vi.distributions.Distribution`.

        Returns
        -------
        None
        """
        mean_name = module.variational_parameter_name(variable, "mean")
        init.constant_(getattr(module, mean_name), self.mean)
        log_std_name = module.variational_parameter_name(variable, "log_std")
        init.constant_(getattr(module, log_std_name), self.log_std)

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor]:
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

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """
        Calculate the log probability of reference given the predictive mean and standard deviation.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        reference: Tensor
            The ground truth label as Tensor of the same shape as each Tensor in
            `parameters`.
        parameters: Tuple[Tensor, Tensor]
            A tuple containing the predictive means and standard deviation as two
            Tensors as returned by :meth:`~predictive_parameters_from_samples`.

        Returns
        -------
        Tensor
            The log probability of the reference under the predicted normal distribution.
            Shape: (1,).
        """
        mean, std = parameters
        variance = std**2
        data_fitting = (reference - mean) ** 2 / variance
        normalization = torch.log(variance)
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)
