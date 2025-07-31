import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.utils.hooks as hooks
from torch import Tensor
from torch.nn import Module, Parameter, init
from torch.nn.common_types import _tensor_list_t
from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_backward_pre_hooks,
    _global_forward_hooks,
    _global_forward_hooks_always_called,
    _global_forward_pre_hooks,
    _WrappedHook,
)

from .priors import MeanFieldNormalPrior
from .utils import NoVariablesError, PostInitCallMeta
from .utils.common_types import VIReturn, _prior_any_t, _vardist_any_t
from .variational_distributions import MeanFieldNormalVarDist


def _forward_unimplemented(self: Module, *input_: Optional[Tensor]) -> VIReturn[Tensor]:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
        For VIModules all inputs must be Tensors.
    """
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class VIModule(Module, metaclass=PostInitCallMeta):
    """
    Base class for Modules using Variational Inference.

    Conceptually, this class takes the place of :class:`torch.nn.Module` for BNNs. It is
    used for any module that has no Bayesian parameters of its own. If the module should
    have Bayesian parameters, use :class:`~.VIBaseModule` instead, which is an extension
    of this class.

    :class:`~.VIModule` contains some additional functionality. Firstly, it keeps track
    of whether the model should return the log probability of the sampled weight (i.e.,
    the log likelihood to obtain these specific values when sampling from the prior or
    variational distribution).These are needed for loss calculation by
    :class:`~.KullbackLeiblerLoss`. If your module contains multiple submodules make
    sure to add all log probabilities and return them, if required by
    :attr:`~self._return_log_probs`. This attribute can be changed with
    :meth:`~return_log_probs`.

    Secondly, a model of nested :class:`~.VIModule` automatically identifies the
    outermost module and sets the :attr:`~self._has_sampling_responsibility` flag. This
    makes the outermost module always accept the keyword argument `samples`, which
    defaults to 10. Since BNNs require multiple samples for each forward pass, the input
    batch is duplicated accordingly and the forward pass is performed vectorized on all
    samples. While this is significantly faster than serial evaluation, it naturally
    requires more memory. Additionally, a certain few operations do not function
    correctly with the vectorization and should not be used in :class:`~.VIModule`.
    Most importantly, this affects the operators ``+=``, ``-=``, ``*=``, and ``/=``.
    However, their longform versions work fine, e.g. ``a = a + b`` instead of ``a += b``.
    """

    forward: Callable[..., VIReturn] = _forward_unimplemented
    random_variables: Optional[Tuple[str, ...]] = None
    _return_log_probs: bool = True
    # _has_sampling_responsibility is set to False right after __init__ completes for
    # each submodule and True for itself that way submodules automatically call forward
    # and the outermost module calls sampled_forward instead
    _has_sampling_responsibility: bool

    def __init__(
        self,
        variable_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        rescale_prior: bool = False,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if variable_shapes is None:
            return
        self.random_variables = tuple(variable_shapes.keys())

        if isinstance(variational_distribution, List):
            assert (
                len(variational_distribution) == len(self.random_variables)
            ), "Provide either exactly one variational distribution or exactly one for each random variable"
            self.variational_distribution = variational_distribution
        else:
            self.variational_distribution = [
                deepcopy(variational_distribution) for _ in self.random_variables
            ]

        if isinstance(prior, List):
            assert (
                len(prior) == len(self.random_variables)
            ), "Provide either exactly one prior distribution or exactly one for each random variable"
            self.prior = prior
        else:
            self.prior = [deepcopy(prior) for _ in self.random_variables]

        if rescale_prior:
            shape_dummy = torch.zeros(variable_shapes[self.random_variables[0]])
            fan_in, _ = init._calculate_fan_in_and_fan_out(shape_dummy)
            for prior in self.prior:
                prior.kaiming_rescale(fan_in)

        self._kaiming_init = kaiming_initialization
        self._rescale_prior = rescale_prior
        self._prior_init = prior_initialization
        self._return_log_probs = return_log_probs

        for variable, vardist in zip(
            self.random_variables, self.variational_distribution
        ):
            assert variable in variable_shapes, f"shape of {variable} is missing"
            shape = variable_shapes[variable]
            for variational_parameter in vardist.variational_parameters:
                parameter_name = self.variational_parameter_name(
                    variable, variational_parameter
                )
                setattr(
                    self,
                    parameter_name,
                    Parameter(torch.empty(shape, **factory_kwargs)),
                )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset or initialize the parameters of the Module."""
        if self.random_variables is None:
            raise NoVariablesError(
                f"{self.__class__.__name__} has no random variables to reset"
            )

        weight_name = self.variational_parameter_name(
            self.random_variables[0],
            self.variational_distribution[0].variational_parameters[0],
        )
        fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self, weight_name))
        for variable, vardist, prior in zip(
            self.random_variables, self.variational_distribution, self.prior
        ):
            vardist.reset_parameters(self, variable, fan_in, self._kaiming_init)
            if self._prior_init:
                prior.reset_parameters(self, variable)

    @staticmethod
    def variational_parameter_name(variable: str, variational_parameter: str) -> str:
        """
        Get the attribute name of the variational parameter for the specified variable.

        Parameters
        ----------
        variable: str
            Random variable name as specified in `self.random_variables`.
        variational_parameter: str
            Variational parameter name as specified by the variational distribution.

        Returns
        -------
        str
            The attribute name of the specified parameter.
        """
        spec = ["", variable, variational_parameter]
        return "_".join(spec)

    def get_variational_parameters(self, variable: str) -> List[Tensor]:
        """
        Get all variational parameters for the specified variable.

        Parameters
        ----------
        variable: str
            Random variable name as specified in `self.random_variables`.

        Returns
        -------
        List[Tensor]
            All variational parameters for the specified variable in the order specified
            by the associated variational distribution.
        """
        if self.random_variables is None:
            raise NoVariablesError(
                f"{self.__class__.__name__} has no variational parameters to get"
            )

        vardist = self.variational_distribution[self.random_variables.index(variable)]
        return [
            getattr(self, self.variational_parameter_name(variable, param))
            for param in vardist.variational_parameters
        ]

    def get_log_probs(self, sampled_params: List[Tensor]) -> Tensor:
        """
        Get prior and variational log prob of the sampled parameters.

        Accepts the sampled parameters as returned by the `self.sample_variables` method
        and calculates the total prior and variational log probability.

        Parameters
        ----------
        sampled_params: List[Tensor]
            Sampled parameters as returned by the `self.sample_variables` method.

        Returns
        -------
        Tensor
            A Tensor containing two values: the the log probability of the sampled
            values, if they were drawn from the prior or variational distribution (in
            that order).
        """
        if self.random_variables is None:
            raise NoVariablesError(f"{self.__class__.__name__} has no random variables")

        device = sampled_params[0].device
        variational_log_prob = torch.tensor([0.0], device=device)
        prior_log_prob = torch.tensor([0.0], device=device)
        for sample, variable, vardist, prior in zip(
            sampled_params,
            self.random_variables,
            self.variational_distribution,
            self.prior,
        ):
            variational_parameters = self.get_variational_parameters(variable)
            variational_log_prob = (
                variational_log_prob
                + vardist.log_prob(sample, *variational_parameters).sum()
            )

            prior_params = [
                getattr(self, self.variational_parameter_name(variable, param))
                for param in prior._required_parameters
            ]
            prior_log_prob = (
                prior_log_prob + prior.log_prob(sample, *prior_params).sum()
            )
        return torch.cat([prior_log_prob, variational_log_prob])

    def sample_variables(self) -> List[Tensor]:
        """
        Draw one sample from the variational distribution of each random variable.

        The variables are returned in the same order as `self.random_variables`.

        Returns
        -------
        List[Tensor]
            The sampled variables.
        """
        if self.random_variables is None:
            raise NoVariablesError(
                f"{self.__class__.__name__} has no random variables to sample"
            )

        params = []
        for variable, vardist in zip(
            self.random_variables, self.variational_distribution
        ):
            variational_parameters = self.get_variational_parameters(variable)
            params.append(vardist.sample(*variational_parameters))
        return params

    @staticmethod
    def _expand_to_samples(input_: Optional[Tensor], samples: int) -> Tensor:
        if input_ is None:
            input_ = torch.tensor(False)
        return input_.expand(samples, *input_.shape)

    def sampled_forward(
        self, *input_: Optional[Tensor], samples: int = 10, **kwargs: Any
    ) -> VIReturn[_tensor_list_t]:
        """
        Forward pass of the module evaluating multiple weight samples.

        This will automatically be called by the outermost module. Instead of the
        :meth:`~forward` method. It grabs the ``samples`` argument, if provided,
        and copies the input batch the specified number of times. The :meth:`~forward`
        is performed vectorized over that additional sample dimension.

        Parameters
        ----------
        input_: Tensor
            Any number of input Tensors
        samples : int, default: 10
            Number of weight samples to evaluate
        kwargs: Any
            Any additional keyword arguments

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...]]
            One or multiple Tensors
        """
        expanded = [self._expand_to_samples(x, samples=samples) for x in input_]
        return torch.vmap(self.forward, randomness="different")(*expanded, **kwargs)

    @property
    def return_log_probs(self) -> bool:
        """
        Set whether the module returns log probabilities.

        Log probabilities are required for most standard losses.

        Returns
        -------
        bool
            Whether the module returns log probabilities.
        """
        return self._return_log_probs

    @return_log_probs.setter
    def return_log_probs(self, mode: bool = True) -> None:
        """
        Set whether the module returns log probabilities.

        Log probabilities are required for most standard losses.

        Parameters
        ----------
        mode : bool, default: True
            Whether to enable (``True``) or disable (``False``) returning of log probs.
        """
        for module in self.modules():
            if isinstance(module, VIModule):
                module._return_log_probs = mode

    def _set_sampling_responsibility(self) -> None:
        for module in self.modules():
            if isinstance(module, VIModule):
                module._has_sampling_responsibility = False
        self._has_sampling_responsibility = True

    def __post_init__(self) -> None:
        """
        After __init__ set sampling responsibility.

        Since higher level modules overwrite _has_sampling_responsibility for lower ones
        only the top level class will have it set to True, making it use sampled_forward
        by default.
        """
        self._set_sampling_responsibility()

    def _pre_forward(self, *input_: Any, **kwargs: Any) -> Any:
        """Select sampled_forward or forward based on _has_sampling_responsibility."""
        if self._has_sampling_responsibility:
            return self.sampled_forward(*input_, **kwargs)
        else:
            return self.forward(*input_, **kwargs)

    # Copied from pytorch 2.4, basically untested since assumed working
    # Change: call _pre_forward instead of forward
    def _slow_forward(self, *input_: Any, **kwargs: Any) -> Any:  # pragma: no cover
        tracing_state = torch._C._get_tracing_state()
        forward_call = self._pre_forward
        if not tracing_state or isinstance(forward_call, torch._C.ScriptMethod):
            return forward_call(*input_, **kwargs)
        recording_scopes = torch.jit._trace._trace_module_map is not None
        if recording_scopes:
            # type ignore was added because at this point one knows that
            # torch.jit._trace._trace_module_map is not Optional and has type Dict[Any, Any]
            name = (
                torch.jit._trace._trace_module_map[self]
                if self in torch.jit._trace._trace_module_map
                else None
            )  # type: ignore[index, operator] # noqa: B950
            if name:
                tracing_state.push_scope(name)
            else:
                recording_scopes = False
        try:
            result = forward_call(*input_, **kwargs)
        finally:
            if recording_scopes:
                tracing_state.pop_scope()
        return result

    # Copied from pytorch 2.4, basically untested since assumed working
    # Change: call _pre_forward instead of forward
    def _call_impl(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        if torch._C._get_tracing_state():
            forward_call = self._slow_forward
        else:
            forward_call = self._pre_forward
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (
            self._backward_hooks
            or self._backward_pre_hooks
            or self._forward_hooks
            or self._forward_pre_hooks
            or _global_backward_hooks
            or _global_backward_pre_hooks
            or _global_forward_hooks
            or _global_forward_pre_hooks
        ):
            return forward_call(*args, **kwargs)

        try:
            result = None
            called_always_called_hooks = set()

            full_backward_hooks, non_full_backward_hooks = [], []
            backward_pre_hooks = []
            if self._backward_pre_hooks or _global_backward_pre_hooks:
                backward_pre_hooks = self._get_backward_pre_hooks()

            if self._backward_hooks or _global_backward_hooks:
                full_backward_hooks, non_full_backward_hooks = (
                    self._get_backward_hooks()
                )

            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
                ):
                    if hook_id in self._forward_pre_hooks_with_kwargs:
                        args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
                        if args_kwargs_result is not None:
                            if (
                                isinstance(args_kwargs_result, tuple)
                                and len(args_kwargs_result) == 2
                            ):
                                args, kwargs = args_kwargs_result
                            else:
                                raise RuntimeError(
                                    "forward pre-hook must return None or a tuple "
                                    f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                                )
                    else:
                        args_result = hook(self, args)
                        if args_result is not None:
                            if not isinstance(args_result, tuple):
                                args_result = (args_result,)
                            args = args_result

            bw_hook = None
            if full_backward_hooks or backward_pre_hooks:
                bw_hook = hooks.BackwardHook(
                    self, full_backward_hooks, backward_pre_hooks
                )
                args = bw_hook.setup_input_hook(args)

            result = forward_call(*args, **kwargs)
            if _global_forward_hooks or self._forward_hooks:
                for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
                ):
                    # mark that always called hook is run
                    if (
                        hook_id in self._forward_hooks_always_called
                        or hook_id in _global_forward_hooks_always_called
                    ):
                        called_always_called_hooks.add(hook_id)

                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)

                    if hook_result is not None:
                        result = hook_result

            if bw_hook:
                if not isinstance(result, (torch.Tensor, tuple)):
                    warnings.warn(
                        "For backward hooks to be called,"
                        " module output should be a Tensor or a tuple of Tensors"
                        f" but received {type(result)}"
                    )
                result = bw_hook.setup_output_hook(result)

            # Handle the non-full backward hooks
            if non_full_backward_hooks:
                var = result
                while not isinstance(var, torch.Tensor):
                    if isinstance(var, dict):
                        var = next(
                            v for v in var.values() if isinstance(v, torch.Tensor)
                        )
                    else:
                        var = var[0]
                grad_fn = var.grad_fn
                if grad_fn is not None:
                    for hook in non_full_backward_hooks:
                        grad_fn.register_hook(_WrappedHook(hook, self))
                    self._maybe_warn_non_full_backward_hook(args, result, grad_fn)

            return result

        except Exception:
            # run always called hooks if they have not already been run
            # For now only forward hooks have the always_call option but perhaps
            # this functionality should be added to full backward hooks as well.
            for hook_id, hook in _global_forward_hooks.items():
                if (
                    hook_id in _global_forward_hooks_always_called
                    and hook_id not in called_always_called_hooks
                ):  # type: ignore[possibly-undefined]
                    try:
                        hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(
                            "global module forward hook with ``always_call=True`` raised an exception "
                            f"that was silenced as another error was raised in forward: {str(e)}"
                        )
                        continue

            for hook_id, hook in self._forward_hooks.items():
                if (
                    hook_id in self._forward_hooks_always_called
                    and hook_id not in called_always_called_hooks
                ):  # type: ignore[possibly-undefined]
                    try:
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)  # type: ignore[possibly-undefined]
                        else:
                            hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(
                            "module forward hook with ``always_call=True`` raised an exception "
                            f"that was silenced as another error was raised in forward: {str(e)}"
                        )
                        continue
            # raise exception raised in try block
            raise


#    """
#    Base class for VIModules that draw weights from a variational distribution.
#
#    Conceptually, this class takes the place of ``torch.nn.Module`` for BNNs.It is used
#    for any module that has Bayesian parameters of its own. If the module should not have
#    Bayesian parameters, use ``VIModule`` instead.
#
#    It dynamically initializes a separate attribute for each parameter of each random
#    variable.
#
#    Random variables are specified in by ``self.random_variables`` and should be set
#    before `super().__init__` is called. Generally, this is any Tensor that would be a
#    ``Parameter`` in pytorch, like the `weight` and `bias` Tensor. The Tensor shape is
#    specified by the argument `variable_shapes`.
#
#    For each random variable a number of attributes are initialized according to the
#    number of entries of `variational_parameters` of the associated
#    :func:`VariationalDistribution<torch_bayesian.vi.base.VariationalDistribution>`.
#
#    The names of these attributes can be discovered using the
#    ``variational_parameter_name`` method.
#
#    This class accepts :func:`VIkwargs<torch_bayesian.vi.VIkwargs>` , but has no default
#    for `variational_distribution` and `prior`.
#
#    Parameters
#    ----------
#    variable_shapes: Dict[str, Tuple[int, ...]]
#        Shape specifications for all random variables. Keys should match the values in
#        `self.random_variables`. Additional keys are ignored.
#    """
