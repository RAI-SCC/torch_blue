from math import log
from typing import Tuple

import torch
from torch import Tensor

from torch_bayesian.vi import _globals

from .base import PredictiveDistribution


class MeanFieldNormalPredictiveDistribution(PredictiveDistribution):
    """
    Predictive distribution assuming uncorrelated, normal distributed forecasts.

    This predictive distribution assumes normal distributed predictions, where all
    outputs are uncorrelated as commonly used for regression tasks.

    When used with :class:`~torch_bayesian.vi.KullbackLeiblerLoss` or
    :class:`~torch_bayesian.vi.AnalyticalKullbackLeiblerLoss` this distribution
    produces a loss related to MSE loss.

    The predictive parameters are the mean and standard deviation of the forecasts.
    """

    predictive_parameters = ("mean", "std")

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
        be set with :func:`~torch_bayesian.vi.utils.use_norm_constants`.

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
