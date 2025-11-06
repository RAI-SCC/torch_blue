from typing import Dict, Optional, Tuple, cast

import torch
from torch import nn

from ..base import VIModule
from ..distributions import MeanFieldNormal
from .common_types import VIkwargs, _dist_any_t
from .init import fixed_

_blacklist = [nn.ReLU, nn.LayerNorm]


def _convert_module(
    module: nn.Module,
    variational_distribution: _dist_any_t = MeanFieldNormal(),
    prior: _dist_any_t = MeanFieldNormal(),
    rescale_prior: bool = False,
    kaiming_initialization: bool = True,
    prior_initialization: bool = False,
    return_log_probs: bool = True,
    keep_weights: bool = False,
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
        device=None,
        dtype=None,
    )
    new_class_name = "AVI" + module.__class__.__name__
    new_class = type(new_class_name, (VIModule, module.__class__), dict())
    setattr(new_class, "forward", module.forward)

    module.__class__ = new_class
    module = cast(VIModule, module)
    if len(module._parameters) == 0:
        VIModule.__post_init__(module)
        return

    parameters = module._parameters
    module._parameters = dict()

    variable_shapes: Dict[str, Optional[Tuple[int, ...]]] = dict()
    variable_types: Dict[str, Tuple[torch.device, torch.dtype]] = dict()
    for name, parameter in parameters.items():
        if parameter is None:
            variable_shapes[name] = None
        else:
            variable_shapes[name] = tuple(parameter.shape)
            variable_types[name] = (parameter.device, parameter.dtype)

    devices = {spec[0] for spec in variable_types.values()}
    dtypes = {spec[1] for spec in variable_types.values()}

    types_set = False
    if len(devices) == 1 and len(dtypes) == 1:
        vikwargs["device"] = devices.pop()
        vikwargs["dtype"] = dtypes.pop()
        types_set = True

    VIModule.__init__(module, variable_shapes, convert_overwrite=True, **vikwargs)
    VIModule.__post_init__(module)

    if not types_set:
        for (var, (device, dtype)), var_dist in zip(
            variable_types.items(), module.variational_distribution
        ):
            for param in var_dist.distribution_parameters():
                param_name = module.variational_parameter_name(var, param)
                getattr(module, param_name).to(device=device, dtype=dtype)

    if keep_weights:
        primary_parameter = variational_distribution.primary_parameter
        for name, parameter in parameters.items():
            if parameter is None:
                continue
            param_name = module.variational_parameter_name(name, primary_parameter)
            fixed_(getattr(module, param_name), parameter)


def convert_to_vimodule(
    module: nn.Module,
    variational_distribution: _dist_any_t = MeanFieldNormal(),
    prior: _dist_any_t = MeanFieldNormal(),
    rescale_prior: bool = False,
    kaiming_initialization: bool = True,
    prior_initialization: bool = False,
    return_log_probs: bool = True,
    keep_weights: bool = False,
) -> None:
    """Convert a PyTorch module to a VIModule."""
    # device and dtype are not present because they are copied on a per-parameter basis
    # from the original modules
    vikwargs = dict(
        variational_distribution=variational_distribution,
        prior=prior,
        rescale_prior=rescale_prior,
        kaiming_initialization=kaiming_initialization,
        prior_initialization=prior_initialization,
        return_log_probs=return_log_probs,
    )
    for m in module.modules():
        _convert_module(m, **vikwargs, keep_weights=keep_weights)

    # __post_init__ needs to be called here again since the loop above goes outside in
    # and __post_init__ relies on being called the last time by the outermost module
    module = cast(VIModule, module)
    VIModule.__post_init__(module)
