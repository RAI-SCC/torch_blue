from typing import TYPE_CHECKING, Optional, Tuple, TypedDict, TypeVar, Union

import torch
from torch import Tensor
from torch.nn.common_types import _scalar_or_tuple_any_t
from typing_extensions import TypeAlias

if TYPE_CHECKING:  # pragma: no cover
    from ..priors import Prior
    from ..variational_distributions import VariationalDistribution

_prior_any_t: TypeAlias = _scalar_or_tuple_any_t["Prior"]
_vardist_any_t: TypeAlias = _scalar_or_tuple_any_t["VariationalDistribution"]

T = TypeVar("T")
_log_prob_return_format = Tuple[T, Tensor]
VIReturn = Union[T, _log_prob_return_format[T]]


class VIkwargs(TypedDict):
    """
    Common keyword arguments for most VIModules.

    Most :class:`~torch_bayesian.vi.VIModule` accept these as keyword arguments. Unless
    specified otherwise, they use the defaults below.

    This class is only used for documentation and type hinting.

    Parameters
    ----------
    variational_distribution: Union[VarDist, List[VarDist]], default: :class:`MeanFieldNormalVarDist()<torch_bayesian.vi.variational_distributions.MeanFieldNormalVarDist>`
        Either one
        :class:`torch_bayesian.vi.variational_distributionss.VariationalDistribution` ,
        which is used for all random variables, or a list of them, one for each random
        variable. This specifies the assumed parametrization of the weight distribution.
    prior: Union[Prior, List[Prior]], default: :class:`MeanFieldNormalPrior()<torch_bayesian.vi.priors.MeanFieldNormalPrior>`
        Either one :class:`~torch_bayesian.vi.priors.Prior` , which is used for all
        random variables, or a list of them, one for each random variable. This
        specifies the previous knowledge about the weight distribution.
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

    variational_distribution: _vardist_any_t
    prior: _prior_any_t
    rescale_prior: bool
    kaiming_initialization: bool
    prior_initialization: bool
    return_log_probs: bool
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]
