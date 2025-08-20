from torch import Tensor, nn

from .base import PredictiveDistribution


class NonBayesianPredictiveDistribution(PredictiveDistribution):
    """
    Predictive distribution for non-Bayesian forecasts.

    This predictive distribution converts sampled predictions into point predictions.
    When used with :class:`~torch_bayesian.vi.KullbackLeiblerLoss` or
    :class:`~torch_bayesian.vi.AnalyticalKullbackLeiblerLoss` this distribution
    allows training with a non-Bayesian loss up to the prior term, which can be disabled
    by setting :attr:`~torch_bayesian.vi.KullbackLeiblerLoss.heat` = 0. It can reproduce
    MSE (L2) and MAE (L1) loss. To reproduce cross-entropy loss, use
    :class:`~.CategoricalPredictiveDistribution` and set
    :attr:`~torch_bayesian.vi.KullbackLeiblerLoss.heat` = 0.

    Parameters
    ----------
    loss_type: str, default: "MSE"
        Sets to loss to imitate. Use "MSE" or "L2" for MSE loss and "MAE" or "L1" for
        MAE loss.

    Raises
    ------
    :exc:`ValueError`
        If the specified loss type is not supported.
    """

    predictive_parameters = ("mean",)

    def __init__(self, loss_type: str = "MSE") -> None:
        super().__init__()
        if loss_type in ["MSE", "L2"]:
            self.loss = nn.MSELoss()
        elif loss_type in ["MAE", "L1"]:
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

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

        This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.

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
        return self.loss(parameters, reference)
