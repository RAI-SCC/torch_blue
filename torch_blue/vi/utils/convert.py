from typing import Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

from torch_blue import vi

from ..base import VIModule
from ..distributions import MeanFieldNormal
from .common_types import VIkwargs, _dist_any_t
from .init import fixed_

__all__ = [
    "convert_to_vimodule",
    "ban_convert",
    "ban_torch_convert",
    "ban_reuse",
]

# Modules in the blacklist will not be converted. The default set contains modules that
# do not have Parameters and therefore would not change when converted. Additionally,
# norms are assumed to be not converted by default.
_blacklist = {
    nn.Identity,
    # activations
    nn.Threshold,
    nn.ReLU,
    nn.RReLU,
    nn.Hardtanh,
    nn.ReLU6,
    nn.Sigmoid,
    nn.Hardsigmoid,
    nn.Tanh,
    nn.SiLU,
    nn.Mish,
    nn.Hardswish,
    nn.ELU,
    nn.CELU,
    nn.SELU,
    nn.GLU,
    nn.GELU,
    nn.Hardshrink,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.Softplus,
    nn.Softshrink,
    nn.Softsign,
    nn.Tanhshrink,
    nn.Softmin,
    nn.Softmax,
    nn.Softmax2d,
    nn.LogSoftmax,
    # distances
    nn.PairwiseDistance,
    nn.CosineSimilarity,
    # dropout
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
    # flatten,
    nn.Flatten,
    nn.Unflatten,
    # fold
    nn.Fold,
    nn.Unfold,
    # padding
    nn.CircularPad1d,
    nn.CircularPad2d,
    nn.CircularPad3d,
    nn.ConstantPad1d,
    nn.ConstantPad2d,
    nn.ConstantPad3d,
    nn.ReflectionPad1d,
    nn.ReflectionPad2d,
    nn.ReflectionPad3d,
    nn.ReplicationPad1d,
    nn.ReplicationPad2d,
    nn.ReplicationPad3d,
    nn.ZeroPad1d,
    nn.ZeroPad2d,
    nn.ZeroPad3d,
    # pixel shuffle
    nn.PixelShuffle,
    nn.PixelUnshuffle,
    # pooling
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.MaxUnpool1d,
    nn.MaxUnpool2d,
    nn.MaxUnpool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.FractionalMaxPool2d,
    nn.FractionalMaxPool3d,
    nn.LPPool1d,
    nn.LPPool2d,
    nn.LPPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    # upsampling
    nn.Upsample,
    nn.UpsamplingNearest2d,
    nn.UpsamplingBilinear2d,
}
_torch_norms = {
    # norms
    nn.BatchNorm1d,
    nn.LazyBatchNorm1d,
    nn.BatchNorm2d,
    nn.LazyBatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.LazyInstanceNorm1d,
    nn.InstanceNorm2d,
    nn.LazyInstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LazyInstanceNorm3d,
    nn.LocalResponseNorm,
    nn.CrossMapLRN2d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.RMSNorm,
}
_blacklist |= _torch_norms
_torch_blacklist = set()
_reuse_blacklist = set()


def convert_norms(mode: bool = True) -> None:
    """
    Set whether to auto-convert PyTorch norm modules.

    Since norms in neural networks are mostly for stability, we assume by default, that
    they should remain non-Bayesian. With `mode=True` this method makes it so norms from
    PyTorch are converted. With `mode=False` it resets to the default behavior.

    Parameters
    ----------
    mode: bool, default=True
        If `True` set auto-conversion to convert PyTorch norms. If `False` reset to
        default behavior of not converting PyTorch norms.

    Returns
    -------
    None

    """
    global _blacklist
    if mode:
        _blacklist.difference_update(_torch_norms)
    else:
        _blacklist.update(_torch_norms)


def ban_convert(
    class_names: Union[Type[nn.Module], List[Type[nn.Module]]], mode: bool = True
) -> None:
    """
    Ban given class names from conversions.

    This method adds one or more class names to the conversion blacklist so they are not
    touched during auto-conversion keeping them non-Bayesian. The default blacklist
    contains a variety to PyTorch modules that do not have parameters and therefore
    would not change during conversion. Additionally, by default PyTorch normalization
    layers are not converted, since their purpose is only stability not learning. If you
    wish to change this behavior use :func:`~convert_norms` to add or remove all PyTorch
    norms from the blacklist.

    Parameters
    ----------
    class_names: Union[str, List[str]]
        A class name or a list of class names not to be converted.
    mode: bool, default=True
        If `False` this will unban the provided names instead.
    """
    global _torch_blacklist
    if isinstance(class_names, type):
        class_names = [class_names]
    if mode:
        _blacklist.update(class_names)
    else:
        _blacklist.difference_update(class_names)


def ban_torch_convert(class_names: Union[str, List[str]], mode: bool = True) -> None:
    """
    Ban given class names from conversions as PyTorch classes.

    As `torch_blue` implements optimized, Bayesian variants of PyTorch modules, classes
    sharing their name are automatically converted to those versions. If you have
    implemented a module that shares the name of a PyTorch module, this method can be
    used to ban Modules with that name from automatic conversion to make it function.

    Parameters
    ----------
    class_names: Union[str, List[str]]
        A class name or a list of class names to not convert like the PyTorch class of
        the same name.
    mode: bool, default=True
        If `False` this will unban the provided names instead.
    """
    global _torch_blacklist
    if isinstance(class_names, str):
        class_names = [class_names]

    if mode:
        _torch_blacklist.update(class_names)
    else:
        _torch_blacklist.difference_update(class_names)


def ban_reuse(class_names: Union[str, List[str]], mode: bool = True) -> None:
    """
    Ban given class names from reuse of the converted class.

    For efficiency all auto-converted classes are stored and reused, if the same class
    name reoccurs. While reuse of the same class name should not occur, this method
    allows to ban reuse of the converted class for the given names to make
    auto-conversion function even if two different module classes with the same name are
    used.

    Parameters
    ----------
    class_names: Union[str, List[str]]
        A class name or a list of class names whose converted versions may not be
        reused.
    mode: bool, default=True
        If `False` this will unban the provided names instead.
    """
    global _reuse_blacklist
    if isinstance(class_names, str):
        class_names = [class_names]

    if mode:
        _reuse_blacklist.update(class_names)
    else:
        _reuse_blacklist.difference_update(class_names)


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
    class_name = module.__class__.__name__
    if (class_name not in _torch_blacklist) and hasattr(vi, "VI" + class_name):
        new_class = getattr(vi, "VI" + class_name)
    elif (class_name not in _reuse_blacklist) and "AVI" + class_name in globals():
        new_class = globals()["AVI" + class_name]
    else:
        new_class_name = "AVI" + class_name
        new_class = type(new_class_name, (VIModule, module.__class__), dict())
        setattr(new_class, "forward", module.__class__.forward)
        globals()[new_class_name] = new_class

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
    module.__class__.__post_init__(module)

    if not types_set:
        for var, (device, dtype) in variable_types.items():
            var_dist = module.variational_distribution[var]
            for param in var_dist.distribution_parameters:
                param_name = module.variational_parameter_name(var, param)
                param = getattr(module, param_name)
                param.data = param.to(device=device, dtype=dtype)

    if keep_weights:
        for name, parameter in parameters.items():
            if parameter is None:
                continue
            var_dist = module.variational_distribution[name]
            primary_parameter = var_dist.primary_parameter
            param_name = module.variational_parameter_name(name, primary_parameter)
            fixed_(getattr(module, param_name), parameter)


def convert_to_vimodule(
    module: nn.Module,
    keep_weights: bool = False,
    variational_distribution: _dist_any_t = MeanFieldNormal(),
    prior: _dist_any_t = MeanFieldNormal(),
    rescale_prior: bool = False,
    kaiming_initialization: bool = True,
    prior_initialization: bool = False,
    return_log_probs: bool = True,
) -> None:
    """
    Convert a PyTorch module to a VIModule.

    This method automatically converts a PyTorch module to a VIModule. This also works
    for any model that is compatible with :meth:`torch.vmap` (documentation
    `here <https://docs.pytorch.org/docs/stable/generated/torch.vmap.html>`__). Usually,
    this will be the case if you do not use the `+=`, `-=`, `*=`, and `/=` operators
    (their long form, i.e. `a = a + b` instead of `a += b` does not cause issues).

    To configure the model the usual :class:`~torch_blue.vi.VIkwargs` can be used,
    except `device` and `dtype` which are copied independently for each weight matrix
    from the original model. This allows to convert a distributed model and maintain
    the distributed structure. This is compatible with `torch.ddp`, but DDP needs to be
    applied after the conversion.

    For Bayesian pretraining or custom initialization schemes `keep_weights` can be set
    to `True`, which will maintain the original weights as value for the primary
    parameter of the weight distribution. While this can vary the primary parameter is
    typically the distribution mean.

    The model is converted inplace so to continue using the original model a copy needs
    to be made before conversion.

    A good method to verify correct model conversion is to set variational distribution
    and prior to :class:`~torch_blue.vi.distributions.NonBayesian` and `keep_weights`
    to `True`. This should make the converted model produce the same results as the
    original. (The output will have an additional sampling dimension and multiple copies
    of the same result unless you pass `samples=1` to the forward call)

    Standard PyTorch layers will automatically be converted to the optimized
    implementation of this library. Therefore, you should try to use class names that
    already exist in PyTorch (like "Transformer"). If you cannot do this, you can use
    :meth:`~.ban_torch_convert` to add classes to a blacklist (or later remove them from
    it) that will make them be converted normally.

    Advanced note: Auto-conversion will ignore many standard modules from PyTorch since
    they do not have weights (making conversion irrelevant) or should not be converted
    (this mostly applies to norm layers).

    While not recommended you can use :meth:`~.convert_norms` to enable or disable
    conversion of all PyTorch norms. Furthermore, you can add (or remove) any class to
    the conversion ban list with :meth:`~.ban_convert`.

    Finally, auto-conversion will try to reuse auto-converted classes. This means if
    you implemented a custom layer type and used it multiple times all instances will
    still be instances of the same (converted) class. To disable this behavior you can
    add (or remove) the original class the reuse ban list with :meth:`~.ban_reuse`.

    Parameters
    ----------
    module: nn.Module
        The Pytorch module to convert.
    keep_weights: bool
        If `True` keep the original weights as value for the primary parameter.
    VIkwargs
        Several standard keyword arguments. See :class:`~.VIkwargs` for details.
        `device` and `dtype` cannot be used since they are copied from the original
        weights.
    """
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

    # __post_init__ needs to be called here again since the loop above goes outside-in
    # and __post_init__ relies on being called the last time by the outermost module
    module = cast(VIModule, module)
    VIModule.__post_init__(module)
