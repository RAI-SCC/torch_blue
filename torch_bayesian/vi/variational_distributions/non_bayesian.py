import torch
from torch import Tensor

from .base import VariationalDistribution


class NonBayesian(VariationalDistribution):
    """
    A variational distribution that turns the network non-Bayesian.

    This variational distribution uses only a mean, which is returned as-is in sampling.
    Therefore, it makes the model forward pass behave like a classical, non-Bayesian
    network.

    To reproduce non-Bayesian training :class:`~torch_bayesian.vi.KullbackLeiblerLoss`
    and :class:`~torch_bayesian.vi.AnalyticalKullbackLeiblerLoss` can be modified to
    reproduce a non-Bayesian loss (see their documentation).
    """

    def __init__(self) -> None:
        super().__init__()
        self.variational_parameters = ("mean",)
        self._default_variational_parameters = (0.0,)

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

    def log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
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
