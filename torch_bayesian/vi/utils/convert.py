from typing import Dict, Optional, Tuple, cast

import torch
from torch import nn

from ..base import VIModule
from ..priors import MeanFieldNormalPrior
from ..variational_distributions import MeanFieldNormalVarDist
from .common_types import VIkwargs, _prior_any_t, _vardist_any_t

_blacklist = [nn.ReLU, nn.LayerNorm]


def _convert_module(
    module: nn.Module,
    variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
    prior: _prior_any_t = MeanFieldNormalPrior(),
    rescale_prior: bool = False,
    kaiming_initialization: bool = True,
    prior_initialization: bool = False,
    return_log_probs: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    if module.__class__ in _blacklist:
        return

    vikwargs: VIkwargs = dict(
        variational_distribution=variational_distribution,
        prior=prior,
        rescale_prior=rescale_prior,
        kaiming_initialization=kaiming_initialization,
        prior_initialization=prior_initialization,
        return_log_probs=return_log_probs,
        device=device,
        dtype=dtype,
    )
    new_class_name = "AVI" + module.__class__.__name__
    new_class = type(new_class_name, (module.__class__, VIModule), dict())

    module.__class__ = new_class
    if len(module._parameters) == 0:
        return

    # TODO: Add weight copying option
    # primary_parameter = variational_distribution.primary_parameter
    parameters = module._parameters
    module._parameters = dict()

    variable_shapes: Dict[str, Optional[Tuple[int, ...]]] = dict()
    for name, parameter in parameters.items():
        if parameter is None:
            variable_shapes[name] = None
        else:
            variable_shapes[name] = tuple(parameter.shape)

    VIModule.__init__(
        cast(VIModule, module), variable_shapes, convert_overwrite=True, **vikwargs
    )


def convert_to_vimodule(
    module: nn.Module,
    variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
    prior: _prior_any_t = MeanFieldNormalPrior(),
    rescale_prior: bool = False,
    kaiming_initialization: bool = True,
    prior_initialization: bool = False,
    return_log_probs: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Convert a PyTorch module to a VIModule."""
    vikwargs: VIkwargs = dict(
        variational_distribution=variational_distribution,
        prior=prior,
        rescale_prior=rescale_prior,
        kaiming_initialization=kaiming_initialization,
        prior_initialization=prior_initialization,
        return_log_probs=return_log_probs,
        device=device,
        dtype=dtype,
    )
    for m in module.modules():
        _convert_module(m, **vikwargs)
    module = cast(VIModule, module)
    module._set_sampling_responsibility()
