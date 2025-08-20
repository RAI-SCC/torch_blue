import torch
from torch import Tensor

from .base import Prior


class UniformPrior(Prior):
    """
    Prior assuming uniformly distributed parameters.

    This prior implies that any parameter is equally likely, i.e. it assumes no prior
    knowledge. While this might seem appropriate for most cases it is often a poor
    choice, since NNs typically have scale invariance in their weights. Therefore, it is
    typically helpful to set a scale by using e.g. a :class:`~.MeanFieldNormalPrior`.

    This prior has no distribution parameters.
    """

    distribution_parameters = ()

    @staticmethod
    def log_prob(sample: Tensor) -> Tensor:
        """
        Compute the log likelihood of a sample based on the prior.

        Since any sample is equally likely, the log likelihood for each is equal with an
        infinite normalization constant. Since this is hardly useful for practical use
        and constants may be offset during training, the log likelihood is always
        returned as zero.

        This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log likelihood for.

        Returns
        -------
        Tensor
            The log likelihood of the sample under the prior, i.e. zero.
        """
        return torch.tensor([0.0], device=sample.device)
