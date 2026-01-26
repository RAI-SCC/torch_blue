from typing import Optional

import torch
from torch import Tensor, nn

from .base import Distribution


class NonBayesian(Distribution):
    """
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

    is_prior = True
    is_variational_distribution = True
    is_predictive_distribution = True
    distribution_parameters = ("mean",)
    mean = None
    _default_variational_parameters = (0.0,)
    _scaling_parameters = ()

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

    @staticmethod
    def prior_log_prob(sample: Tensor) -> Tensor:
        """
        Compute the log probability of a sample based on the prior.

        Since any sample is equally likely, the log probability for each is equal with an
        infinite normalization constant. Since this is hardly useful for practical use
        and constants may be offset during training, the log probability is always
        returned as zero.

        This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log probability for.

        Returns
        -------
        Tensor
            The log probability of the sample under the prior, i.e. zero.
        """
        return torch.tensor([0.0], device=sample.device)

    def variational_log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
        """
        Return 0 as dummy log probability.

        Dummy log_prob that returns 0.

        Parameters
        ----------
        sample: Tensor
            The current weight configuration.
        mean: Tensor
            The current weight values. Usually this should be the same as `sample`, but
            this is not enforce.

        Returns
        -------
        Tensor
            A Tensor of zeroes the same shape as `sample`.
        """
        return torch.zeros_like(sample)

    def sample(self, mean: Tensor) -> Tensor:
        """
        Return input as sample.

        Dummy sample that returns mean.

        Parameters
        ----------
        mean: Tensor
            The current weight values.

        Returns
        -------
        Tensor
            The unchanged weight values.
        """
        return mean

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tensor:
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

    def log_prob_from_parameters(self, reference: Tensor, parameters: Tensor) -> Tensor:
        """
        Calculate the loss of the mean prediction with respect to reference.

        Since the loss works on NEGATIVE log likelihood this is minus the specified
        loss. This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.

        Parameters
        ----------
        reference: Tensor
            The ground truth label as Tensor of the same shape as `parameters`.
        parameters: Tensor
            The predictive means as Tensor of shape as `reference` and as returned by
            :meth:`~predictive_parameters_from_samples`.

        Returns
        -------
        Tensor
            The loss of the prediction with respect to the reference. Shape: (1,).
        """
        if self.loss is None:
            raise ValueError("loss_type must be set during initialization")
        return -self.loss(parameters, reference)


class UniformPrior(NonBayesian):
    """
    Alias for :class:`.NonBayesian` that disables variational and predictive settings.

    While this class has the same functionality as :class:`.NonBayesian`, it disables
    the flags for variational and predictive settings.

    It is intended for readability, while trying to avoid incorrect usage since it does
    not represent the behavior of a uniform predictive or variational distribution.
    """

    is_variational_distribution = False
    is_predictive_distribution = False
