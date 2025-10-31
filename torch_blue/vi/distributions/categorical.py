import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .base import Distribution


class Categorical(Distribution):
    """
    Categorical distribution used as predictive distribution for classification tasks.

    This distribution is implemented only as a predictive distribution.
    It interprets the model output either as logits (default) or as probabilities. In
    both cases, the distribution parameters are class probabilities with their sum
    normalized to one across classes. The last dimension of the model output is
    interpreted as the number of classes.

    When used with :class:`~torch_blue.vi.KullbackLeiblerLoss` or
    :class:`~torch_blue.vi.AnalyticalKullbackLeiblerLoss` this distribution
    produces a loss related to cross-entropy loss.

    Parameters
    ----------
    input_type: str, default: "logits"
        Whether to interpret the model output as logits ("logits"; i.e., log
        probabilities) or probabilities ("probs").

    Raises
    ------
    :exc:`AssertionError`
        If ``input_type`` is neither "logits" nor "probs".
    """

    is_prior = False
    is_variational_distribution = False
    is_predictive_distribution = True

    def __init__(self, input_type: str = "logits"):
        self.distribution_parameters = ("probs",)
        assert input_type in ["logits", "probs"], "input_type must be logits or probs"
        self._in_logits = input_type == "logits"

    def predictive_parameters_from_samples(
        self, samples: Tensor, eps: float = 1e-5
    ) -> Tensor:
        """
        Calculate predictive probabilities from samples.

        Converts logits to probabilities if ``input_type`` is "logits" and normalizes
        along the class dimension.

        Parameters
        ----------
        samples: Tensor
            The model output as Tensor of shape (S, B, C), where S is the number of
            samples, B is the batch size, and C is the number of classes. The batch size
            dimension is optional.
        eps: float, default: 1e-5
            Epsilon for numerical stability.

        Returns
        -------
        Tensor
            The predictive class probabilities as Tensor of shape (B, C).
        """
        if self._in_logits:
            return F.softmax(samples + eps, -1).mean(dim=0)
        else:
            normalized = samples / samples.sum(dim=-1, keepdim=True)
            return normalized.mean(dim=0)

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tensor, eps: float = 1e-5
    ) -> Tensor:
        """
        Calculate the log probability of the label based on the class probabilities.

        This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.

        Parameters
        ----------
        reference: Tensor
            The ground truth label as Tensor of shape (B,), where B is the batch size
            and may be one.
        parameters: Tensor
            The predictive class probabilities as Tensor of shape (B, C) as returned by
            :meth:`~predictive_parameters_from_samples`.
        eps: float, default: 1e-5
            Epsilon for numerical stability.

        Returns
        -------
        Tensor
            The log probability of the label under the predicted class probabilities.
            Shape: (1,).
        """
        parameters = torch.log(parameters + eps)
        value = reference.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, parameters)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)
