from typing import Optional

import torch
from torch import Tensor, nn

from .base import PredictiveDistribution, Prior, VariationalDistribution


class UniformPrior(Prior):
    """
    A uniform prior, that gives equal weight to all values.

    While this might seem like a good choice for an unknown prior it typically gives too
    much weight to larger weight values and a :class:`~.MeanFieldNormal` prior is typically
    preferable. However, it can be used to imitate nom-Bayesian behavior and is
    equivalent to :class:`~.NonBayesian` used as prior.
    """

    distribution_parameters = ("mean",)
    mean = None
    _scaling_parameters = ()

    def log_prob(self, sample: Tensor, parameters: tuple[Tensor]) -> Tensor:
        r"""
        Return 0 as dummy log probability.

        Dummy log_prob that returns 0.

        Parameters
        ----------
        sample: Tensor
            The current weight configuration.
        parameters: tuple[Tensor]
            The current parameter values.

        Returns
        -------
        Tensor
            A Tensor of zeroes the same shape as `sample`.
        """
        return torch.zeros_like(sample)


class NonBayesian(UniformPrior, VariationalDistribution, PredictiveDistribution):
    r"""
    Pseudo-distribution that imitates non-Bayesian behavior.

    This distribution is implemented as prior, variational distribution, and predictive
    distribution. Its distribution parameter is "mean" representing a fixed value.

    As prior it functions as a uniform prior.

    As variational distribution it causes fixed weights.

    As predictive distribution it imitates a non-Bayesian loss, which needs to be
    specified during initialization.

    Parameters
    ----------
    loss_type: Optional[str], default = None
        Type of loss function to be used. Available options are:
        MAE, L1, MSE, L2

    Raises
    ------
    ValueError
        If loss_type is not supported.
    """

    _default_variational_parameters = (0.0,)

    def __init__(self, loss_type: Optional[str] = None) -> None:
        super().__init__()
        if loss_type is None:
            self.loss = None
        elif loss_type in ["MSE", "L2"]:
            self.loss = nn.MSELoss()
        elif loss_type in ["MAE", "L1"]:
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def sample(self, parameters: tuple[Tensor]) -> Tensor:
        r"""
        Return input as sample.

        Dummy sample that returns mean.

        Parameters
        ----------
        parameters: tuple[Tensor]
            The current parameter values.

        Returns
        -------
        Tensor
            The unchanged weight values.
        """
        mean = parameters[0]
        return mean

    def predictive_parameters_from_samples(self, samples: Tensor) -> Tensor:
        r"""
        Calculate predictive mean from samples.

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
        """
        return samples.mean(dim=0)

    def log_prob_from_parameters(
        self, reference: Tensor, parameters: tuple[Tensor]
    ) -> Tensor:
        r"""
        Calculate the loss of the mean prediction with respect to reference.

        Since the loss works on NEGATIVE log likelihood this is minus the specified
        loss. This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.

        Parameters
        ----------
        reference: Tensor
            The ground truth label as Tensor of the same shape as `parameters`.
        parameters: tuple[Tensor]
            The current parameter values.

        Returns
        -------
        Tensor
            The loss of the prediction with respect to the reference. Shape: (1,).
        """
        if self.loss is None:
            raise ValueError("loss_type must be set during initialization")
        return -self.loss(parameters, reference)
