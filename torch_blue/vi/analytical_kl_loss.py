from abc import ABC
from math import log
from typing import Callable, Dict, Iterable, List, Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module

from . import _globals
from .base import VIModule
from .distributions import (
    MeanFieldNormal,
    NonBayesian,
    PredictiveDistribution,
    Prior,
    UniformPrior,
    VariationalDistribution,
)
from .utils import UnsupportedDistributionError


def _forward_unimplemented(
    self: "KullbackLeiblerModule", *input_: Optional[Tensor]
) -> Tensor:
    """
    Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within this function,
        one should call the :class:`KullbackLeiblerModule` instance afterward instead of
        this.
    """
    raise NotImplementedError(
        f'{type(self).__name__} is missing the required "forward" function'
    )


class KullbackLeiblerModule(ABC):
    """
    Base class for modules calculating the Kullback-Leibler divergence from distribution parameters.

    A KullbackLeiblerModule calculates the analytical Kullback-Leibler divergence
    between a :class:`~.distributions.Prior` and a
    :class:`~.distributions.VariationalDistribution` based on their parameters. They are
    mainly intended for use with the :class:`~.AnalyticalKullbackLeiblerLoss`.

    Each subclass must define a forward function that is passed, as positional arguments,
    the parameters of the :class:`~.distributions.Prior` in the order specified in its
    :attr:`~.distributions.Distribution.distribution_parameters` attribute followed by
    the parameters of the :class:`~.distributions.VariationalDistribution` in the order
    specified in its :attr:`~.distributions.Distribution.distribution_parameters`
    attribute.
    """

    forward: Callable[..., Tensor] = _forward_unimplemented

    def __call__(
        self,
        prior_parameters: Iterable[Tensor],
        variational_parameters: Iterable[Tensor],
    ) -> Tensor:
        """Distribute parameters to forward function."""
        return self.forward(*prior_parameters, *variational_parameters)


class NormalNormalDivergence(KullbackLeiblerModule):
    """
    Kullback-Leibler divergence between two normal distributions.

    Calculates the KL-Divergence between a :class:`~.distributions.MeanFieldNormal` and
    a :class:`~.distributions.MeanFieldNormal`.
    """

    @staticmethod
    def forward(
        prior_mean: Union[Tensor, float],
        prior_log_std: Union[Tensor, float],
        variational_mean: Tensor,
        variational_log_std: Tensor,
    ) -> Tensor:
        """
        Calculate the Kullback-Leibler divergence.

        All input tensors must have the same shape.

        This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.

        Parameters
        ----------
        prior_mean: Union[Tensor, float]
            Means of the prior distribution.
        prior_log_std: Union[Tensor, float]
            Log standard deviations of the prior distribution.
        variational_mean: Tensor
            Means of the variational distribution.
        variational_log_std: Tensor
            Standard deviations of the variational distribution.

        Returns
        -------
        Tensor
            The KL-Divergence of the two distributions.
        """
        variational_variance = torch.exp(2 * variational_log_std)
        prior_variance = torch.exp(
            torch.tensor(2, device=variational_mean.device) * prior_log_std
        )
        variance_ratio = prior_variance / variational_variance

        raw_kl = (
            variance_ratio.log()
            + (prior_mean - variational_mean).pow(2) / prior_variance
            + 1 / variance_ratio
            - 1
        ) / 2
        return raw_kl.sum()


class NonBayesianDivergence(KullbackLeiblerModule):
    """
    Placeholder Kullback-Leibler divergence for non-Bayesian models.

    This module can be used to disable the prior matching term of the
    :class:`~.AnalyticalKullbackLeiblerLoss`. Together with a
    :class:`~.distributions.NonBayesian` predictive distribution it yields a
    non-Bayesian loss.
    """

    @staticmethod
    def forward(*args: Tensor) -> Tensor:
        """
        Return placeholder zero.

        At least one argument of type Tensor is expected, since the device of the last
        one is used to define the device of the output Tensor.

        This is not affected by :data:`_globals._USE_NORM_CONSTANTS`.
        """
        return torch.tensor([0.0], device=args[-1].device)


class UniformNormalDivergence(KullbackLeiblerModule):
    """
    Kullback-Leibler divergence between a uniform and normal distribution.

    Calculates the KL-Divergence between a :class:`~.distributions.UniformPrior` and a
    :class:`~.distributions.MeanFieldNormal` distribution.
    """

    @staticmethod
    def forward(
        prior_mean: None, variational_mean: Tensor, variational_log_std: Tensor
    ) -> Tensor:
        """
        Calculate the Kullback-Leibler divergence.

        All input tensors must have the same shape.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        prior_mean: None
            Always None, exists for technical reasons.
        variational_mean: Tensor
            Means of the variational distribution.
        variational_log_std: Tensor
            Standard deviations of the variational distribution.

        Returns
        -------
        Tensor
            The KL-Divergence of the two distributions.
        """
        raw_kl = -variational_log_std - 0.5
        if _globals._USE_NORM_CONSTANTS:
            raw_kl = raw_kl - 0.5 * log(2 * torch.pi)
        return raw_kl.sum()


_kl_div_dict: Dict[str, Type[KullbackLeiblerModule]] = dict(
    NormalNormalDivergence=NormalNormalDivergence,
    UniformNormalDivergence=UniformNormalDivergence,
)


class AnalyticalKullbackLeiblerLoss(Module):
    """
    Analytical Kullback-Leibler loss function.

    A version of the Kullback-Leibler loss function that calculates the prior matching
    term analytically from the prior and variational parameters. To that end it stores a
    reference to the model for access to the parameters. Furthermore, only specific
    combinations of :class:`~.distributions.Prior` and
    :class:`~.distributions.VariationalDistribution` are supported (see table below).
    Additionally, it can emulate a non-Bayesian loss, when provided a model with a
    :class:`~.distributions.NonBayesian` variational distribution and a
    :class:`~.distributions.NonBayesian` prior.

    .. list-table:: Supported class combinations
        :widths: 33 33 33
        :header-rows: 1

        * - Prior
          - Variational distribution
          - Kullback-Leibler module
        * - :class:`~.distributions.MeanFieldNormal`
          - :class:`~.distributions.MeanFieldNormal`
          - :class:`~.NormalNormalDivergence`
        * - :class:`~.distributions.UniformPrior`
          - :class:`~.distributions.MeanFieldNormal`
          - :class:`~.UniformNormalDivergence`
        * - :class:`~.distributions.UniformPrior`
          - :class:`~.distributions.NonBayesian`
          - :class:`~.NonBayesianDivergence`

    Parameters
    ----------
    model: :class:`~.VIModule`
        The model to be trained.
    predictive_distribution: :class:`~.distributions.PredictiveDistribution`
        The kind of distribution to assume for the forecasts. This is closely related to
        the non-Bayesian losses, e.g. :class:`~.distributions.MeanFieldNormal`
        corresponds to MSE loss, while :class:`~.distributions.Categorical`
        corresponds to cross-entropy loss.
    dataset_size: Optional[int], default: None
        Number of samples in the training dataset. If not provided here it must be
        provided during the forward call (which also takes precedence if both are
        provided).
    divergence_type: Optional[:class:`~.KullbackLeiblerModule`], default: None
        If ``None`` the correct :class:`~.KullbackLeiblerModule` is selected from the
        integrated options, if available. Otherwise, this is used to manually specify
        the used :class:`~.KullbackLeiblerModule`, typically to integrate custom modules.
    heat: float, default: 1.0
        A factor multiplied with the prior matching term. Set smaller than 1. to produce
        a "cold posterior" loss. Set to 0. to disable the prior matching term to imitate
        a non-Bayesian loss.
    track: bool, default: False
        If ``True`` the loss components are tracked for every forward pass in
        :attr:`~.AnalyticalKullbackLeiblerLoss.log`. This can be enabled, disabled and
        re-enable with the :meth:`~.AnalyticalKullbackLeiblerLoss.track` method. Any
        stored data will remain even if disabled and re-enabled.

    Attributes
    ----------
    log: Optional[Dict[str, List[Tensor]]]
        If tracking was never enabled this is ``None``. Otherwise, it is a dictionary
        with the keys `data_fitting` and `prior_matching`. These two components are
        appended every forward pass while tracking is enabled. Their sum is the total
        loss.

    Raises
    ------
    :exc:`AssertionError`:
        If ``divergence_type`` is ``None`` and prior and variational distribution
        are not consistent in all submodules, since this is not supported yet.
    :exc:`NotImplementedError`:
        If ``divergence_type`` is ``None`` and the combination of prior and variational
        distribution is not supported.
    :exc:`ValueError`:
        If ``divergence_type`` is ``None`` and the model does not contain any
        :class:`~.VIModule`, i.e. is non-Bayesian.
    :exc:`UnsupportedDistributionError`:
        If ``predictive_distribution`` does not support being use as predictive
        distribution.
    """

    def __init__(
        self,
        model: VIModule,
        predictive_distribution: PredictiveDistribution,
        dataset_size: Optional[int] = None,
        divergence_type: Optional["KullbackLeiblerModule"] = None,
        heat: float = 1.0,
        track: bool = False,
    ) -> None:
        super().__init__()
        self.predictive_distribution = predictive_distribution
        self.dataset_size = dataset_size
        self.heat = heat
        self._track = track

        if not isinstance(predictive_distribution, PredictiveDistribution):
            raise UnsupportedDistributionError(
                f"{predictive_distribution.__class__.__name__} does not support use as"
                f" predictive distribution"
            )

        if divergence_type is None:
            for module in model.modules():
                if (
                    not hasattr(module, "random_variables")
                ) or module.random_variables is None:
                    continue
                for prior, var_dist in zip(
                    module.prior.values(), module.variational_distribution.values()
                ):
                    if prior is None or var_dist is None:
                        continue
                    kl_type = self._detect_divergence(prior, var_dist)
                    if divergence_type is None:
                        divergence_type = kl_type
                    else:
                        try:
                            assert isinstance(kl_type, type(divergence_type))
                        except AssertionError:
                            raise NotImplementedError(
                                "Handling of inconsistent distributions types is not "
                                "implemented yet."
                            )
        if divergence_type is None:
            raise ValueError("Provided model is not bayesian.")

        self.kl_module: KullbackLeiblerModule = divergence_type
        model.return_log_probs = False
        self.model = model

        self.log: Optional[Dict[str, List[Tensor]]] = None
        if self._track:
            self._init_log()

    def track(self, mode: bool = True) -> None:
        """
        Enable or disable loss tracking.

        Any existing loss history is kept and continued if tracking is re-enabled.

        Parameters
        ----------
        mode: bool, default: True
            If ``True``, enable loss tracking if ``False`` disable it.

        Returns
        -------
        None
        """
        if mode and self.log is None:
            self._init_log()
        self._track = mode

    def _init_log(self) -> None:
        self.log = dict(data_fitting=[], prior_matching=[])

    @staticmethod
    def _detect_divergence(
        prior: Prior, var_dist: VariationalDistribution
    ) -> KullbackLeiblerModule:
        if isinstance(prior, MeanFieldNormal):
            prior_name = "Normal"
        elif isinstance(prior, UniformPrior):
            prior_name = "Uniform"
        else:
            prior_name = None
        if isinstance(var_dist, NonBayesian):
            return NonBayesianDivergence()
        elif isinstance(var_dist, MeanFieldNormal):
            vardist_name = "Normal"
        else:
            vardist_name = None

        if (prior_name is None) or (vardist_name is None):
            raise NotImplementedError(
                f"Analytical loss is not implemented for {prior.__class__.__name__} and"
                f" {var_dist.__class__.__name__}."
            )

        return _kl_div_dict[prior_name + vardist_name + "Divergence"]()

    def _get_flat_params(self) -> tuple[Tensor, ...]:
        """Extract all parameters as tensors for vectorized operations."""
        prior_param_lol: List[List[Tensor]] = []
        variational_param_lol: List[List[Tensor]] = []

        for module in self.model.modules():
            if (
                not hasattr(module, "random_variables")
            ) or module.random_variables is None:
                continue

            for var, prior in zip(module.random_variables, module.prior.values()):
                if prior is None:
                    continue

                # Extract prior parameters efficiently
                prior_params = [
                    getattr(prior, param) for param in prior.distribution_parameters
                ]

                var_params = module.get_variational_parameters(var)
                for i, var_param in enumerate(var_params):
                    if len(variational_param_lol) < i + 1:
                        variational_param_lol.append([])
                    variational_param_lol[i].append(var_param.flatten())

                # Get number of elements in the last variational parameter
                n_elements = variational_param_lol[0][-1].shape[0]
                # Get device from the last variational parameter
                device = variational_param_lol[0][-1].device
                # Get dtype from the last variational parameter
                dtype = variational_param_lol[0][-1].dtype

                # Broadcast scalar parameters to match dimensions using tensor ops
                for i, prior_param in enumerate(prior_params):
                    if len(prior_param_lol) < i + 1:
                        prior_param_lol.append([])
                    prior_param_lol[i].append(
                        torch.full(
                            (n_elements,),
                            prior_param if prior_param else 0.0,
                            dtype=dtype,
                            device=device,
                        )
                    )

        # Concatenate all tensors at once (more efficient than extending)
        return (
            *(torch.cat(prior_param_list) for prior_param_list in prior_param_lol),
            *(
                torch.cat(variational_param_list)
                for variational_param_list in variational_param_lol
            ),
        )

    def prior_matching(self) -> Tensor:
        """
        Calculate the prior matching KL-Divergence of :attr:`~self.model`.

        Returns
        -------
        Tensor
            The prior matching KL-Divergence of :attr:`~self.model`.
        """
        return self.kl_module.forward(*self._get_flat_params())

    def forward(
        self, model_output: Tensor, target: Tensor, dataset_size: Optional[int] = None
    ) -> Tensor:
        r"""
        Calculate the negative ELBO loss from sampled evaluations and a target.

        Accepts a Tensor of N samples and a target to calculate the loss.

        Parameters
        ----------
        model_output: Tensor
            The model output with ``model.return_log_probs`` = ``False``, i.e. the
            sampled model prediction. Shape: (N, \*)
        target: Tensor
            Target prediction. Shape (\*)
        dataset_size: Optional[int], default: None
            Total number of samples in the dataset. Used in place of
            :attr:`~self.dataset_size` if provided. Must be specified if
            :attr:`~self.dataset_size` is ``None``.

        Returns
        -------
        Tensor
            Negative ELBO loss. Shape: (1,)
        """
        samples = model_output

        if (dataset_size is None) and (self.dataset_size is None):
            raise ValueError(
                "dataset_size must be provided either in the constructor or in the "
                "forward method. It is required to correctly balance the data fitting "
                "and prior matching terms of the ELBO loss."
            )
        n_data = dataset_size or self.dataset_size

        prior_matching = self.heat * self.prior_matching()
        # Sample average for predictive log prob is already done
        data_fitting = (
            -n_data
            * self.predictive_distribution.log_prob_from_samples(target, samples)
            .mean(0)
            .sum()
        )

        if self._track and self.log is not None:
            self.log["data_fitting"].append(data_fitting.item())
            self.log["prior_matching"].append(prior_matching.item())

        return data_fitting + prior_matching
