from typing import TYPE_CHECKING, Optional, TypedDict

import torch
from torch.nn.common_types import _scalar_or_tuple_any_t
from typing_extensions import TypeAlias

if TYPE_CHECKING:  # pragma: no cover
    from ..distributions import Distribution

_dist_any_t: TypeAlias = _scalar_or_tuple_any_t["Distribution"]


class VIkwargs(TypedDict):
    """
    Common keyword arguments for most VIModules.

    Most :class:`~torch_bayesian.vi.VIModule` accept these as keyword arguments. Unless
    specified otherwise, they use the defaults below.

    This class is only used for documentation and type hinting.

    Parameters
    ----------
    variational_distribution: Union[Distribution, List[Distribution]], default: :class:`MeanFieldNormal()<torch_bayesian.vi.distributions.MeanFieldNormal>`
        Either one :class:`torch_bayesian.vi.distributions.Distribution` ,
        which is used for all random variables, or a list of them, one for each random
        variable. This specifies the assumed parametrization of the weight distribution.
    prior: Union[Distribution, List[Distribution]], default: :class:`MeanFieldNormal()<torch_bayesian.vi.distributions.MeanFieldNormal>`
        Either one :class:`~torch_bayesian.vi.distributions.Distribution` , which is
        used for all random variables, or a list of them, one for each random variable.
        This specifies the previous knowledge about the weight distribution.
    rescale_prior: bool, default: False
        If ``True`` , the priors :attr:`_scaling_parameters` are scaled with the sqrt of
        the layer width. This may be necessary to maintain normalization for wide
        layers.
    prior_initialization: bool, default: False
        If ``True`` parameters are initialized according to the prior. If ``False``
        parameters are initialized similar to non-Bayesian networks.
    return_log_probs: bool, default: True
        If ``True`` the model forward pass returns the log probability of the sampled
        weights. This is required for use of
        :class:`~torch_bayesian.vi.KullbackLeiblerLoss`.
    device: Optional[torch.device], default: None
        The torch.device on which the module should be stored.
    dtype: Optional[torch.dtype], default: None
        The torch.dtype of the module parameters.
    """

    variational_distribution: _dist_any_t
    prior: _dist_any_t
    rescale_prior: bool
    kaiming_initialization: bool
    prior_initialization: bool
    return_log_probs: bool
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]
