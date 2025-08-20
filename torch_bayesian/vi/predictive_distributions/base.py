from typing import Callable, Tuple, Union

from torch import Tensor

from ..utils import PostInitCallMeta


class PredictiveDistribution(metaclass=PostInitCallMeta):
    r"""
    Base class for all predictive distributions.

    A predictive distribution is the assumed distribution of the model outputs. Its
    parameters should be derivable from sufficient samples for the same prediction.
    Each distribution must define which parameters are used to represent a prediction.
    For example, regression might use a predictive mean and standard deviation, while
    classification might use a probability for each class.

    Furthermore, the distribution must be able to assign a probability to each possible
    prediction given the expected prediction. This is required for loss calculation.
    Typically, it is enough for subclasses to define :meth:`~log_prob_from_parameters`
    and :meth:`~predictive_parameters_from_samples`, which the class automatically uses
    to first calculate the predictive parameters from the provided samples and then the
    log likelihood of those samples from the parameters. In case this detour does not
    work :meth:`~log_prob_from_samples` can be overwritten. However,
    :meth:`~predictive_parameters_from_samples` should still be defined to allow
    extracting predictions.

    Parameters
    ----------
    predictive_parameters: Tuple[str, ...]
        String names of the predictive parameters. Mainly for documentation purposes.

    Methods
    -------
    predictive_parameters_from_samples: Callable[[Tensor], Union[Tensor, Tuple[Tensor, ...]]]
        Abstract method that accepts the output of a model as Tensor of shape (S, \*),
        where S is the number of samples. Calculates the predictive parameters implied
        by the samples.

    Parameters
    ----------
        samples: Tensor
            The model output as Tensor of shape (S, B, \*), where S is the number of
            samples and B is the batch size.

    Returns
    -------
        Union[Tensor, Tuple[Tensor, ...]]
            One Tensor for each predictive parameter in the order specified in
            :attr:`~self.predictive_parameters`.

    log_prob_from_parameters: Callable[[Tensor, Union[Tensor, Tuple[Tensor, ...]]], Tensor]
        Abstract method that accepts a reference and the predictive parameters as
        calculated by :meth:`~predictive_parameters_from_samples` and calculates the log
        likelihood of the reference under the predicted distribution.

    Parameters
    ----------
        reference: Tensor
            The ground truth or label as Tensor of shape (B, \*), where B is the batch
            size.
        parameters: Union[Tensor, Tuple[Tensor, ...]]
            The parameters specifying the predicted distribution in the order specified
            in :attr:`~self.predictive_parameters`.

    Returns
    -------
        Tensor
            The log likelihood of the reference under the predicted distribution.
            Shape: (1,).
    """

    predictive_parameters: Tuple[str, ...]
    predictive_parameters_from_samples: Callable[
        [Tensor], Union[Tensor, Tuple[Tensor, ...]]
    ]
    log_prob_from_parameters: Callable[
        [Tensor, Union[Tensor, Tuple[Tensor, ...]]], Tensor
    ]

    def __post_init__(self) -> None:
        """Ensure the instance has all required attributes."""
        if not hasattr(self, "predictive_parameters"):
            raise NotImplementedError("Subclasses must define predictive_parameters")
        if not hasattr(self, "predictive_parameters_from_samples"):
            raise NotImplementedError(
                "Subclasses must define predictive_parameters_from_samples"
            )
        if not hasattr(self, "log_prob_from_parameters"):
            raise NotImplementedError("Subclasses must define log_prob_from_parameters")

    def log_prob_from_samples(self, reference: Tensor, samples: Tensor) -> Tensor:
        r"""
        Calculate the log likelihood for reference given a set of samples.

        Usually combines :meth:`~predictive_parameters_from_samples` and
        :meth:`~log_prob_from_parameters`, but can be redefined, if needed.

        Parameters
        ----------
        reference : Tensor
            Expected prediction as Tensor of shape (\*)
        samples : Tensor
            Model prediction as Tensor of shape (S, \*), where S is the number of samples.

        Returns
        -------
        Tensor
            The log likelihood of the reference under the predicted distribution.
            Shape: (1,).
        """
        params = self.predictive_parameters_from_samples(samples)
        return self.log_prob_from_parameters(reference, params)
