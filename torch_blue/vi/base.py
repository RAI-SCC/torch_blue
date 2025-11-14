import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.utils.hooks as hooks
from torch import Tensor
from torch._C._functorch import get_unwrapped
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

from .distributions import MeanFieldNormal
from .utils import NoVariablesError, PostInitCallMeta, UnsupportedDistributionError
from .utils.common_types import _dist_any_t
from .utils.vi_return import VIReturn


def _forward_unimplemented(self: Module, *input_: Optional[Tensor]) -> Tensor:
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

    This class takes the place of :class:`torch.nn.Module` for BNNs. It is used for any
    Bayesian module. While a :class:`torch.nn.Module` may be contained within a
    :class:`~.VIModule` and vice versa there should always be a singular
    :class:`~.VIModule` containing all others to avoid superfluous sampling dimensions.

    :class:`~.VIModule` contains some additional functionality. Firstly, it keeps track
    of whether the model should return the log probability of the sampled weight (i.e.,
    the log probability to obtain these specific values when sampling from the prior or
    variational distribution).These are needed for loss calculation by
    :class:`~.KullbackLeiblerLoss`. This is handled automatically as part of accessing
    the weight matrices, if :attr:`~self.return_log_probs` is True. For evaluation, it
    can be set to `False`.

    The outermost module will aggregate the log probabilities and pack the output and
    log probs into a :class:`~.VIReturn` object, which behaves like a pytorch Tensor.
    However, it has the added attribute :attr:`log_probs` where the log probs are
    stored. This is used by losses, therefore it is easiest to wrap all operations into
    a :class:`~.VIModule` and feed the output directly into a loss function. Classical
    losses will treat it like a Tensor and `torch_blue` losses can use the log prob
    information. If the model has multiple output Tensors, each will contain the full
    log prob information.

    .. IMPORTANT:: When defining custom modules with weights make sure to retrieve them
        using :meth:`~self.sample_variables` as this will maintain the automatic log
        prob tracking.

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

    If the constructed module does not have its own weights, :meth:`super().__init__()`
    is called without arguments. In this setting the methods :meth:`get_log_probs`,
    :meth:`get_variational_parameters`, :meth:`reset_variational_parameters`, and
    :meth:`sample_variable` cannot be used and raise
    :exc:`~torch_blue.vi.utils.NoVariablesError`.

    Any weight matrix in a BNN may require multiple parameters (e.g. mean and std).
    These are stored in separate attributes and therefore cannot be created without
    knowledge of the variational distribution. Therefore, only their shapes are provided
    to :meth:`super().__init__()` via the `variable_shapes` argument. This should be a
    dictionary, where the keys are the random variable names as strings (e.g. "weight"
    and "bias") and the values are tuples of integers specifying the shape.

    After initialization these variables can be accessed as an attribute under the
    specified name. Each access will yield a new sample. The shape of a random variable
    can be set to ``None``. In that case accessing it will always return ``None``.

    .. NOTE:: The insertion order of the dictionary becomes the order
        :attr:`self.random_variables`.

    The names of the created attributes can be discovered using the
    :meth:`~self.variational_parameter_name()` method.

    Additionally, a module with random variables accepts arguments from
    :class:`~.VIkwargs` as keyword arguments. If a list of priors or variational
    distributions is provided they are again assumed to follow the insertion order as
    described above.

    Parameters
    ----------
    variable_shapes: Optional[Dict[str, Optional[Tuple[int, ...]]]], default = None
        Shape specifications for all random variables. Keys are turned into
        :attr:`self.random_variables` in insertion order.
    VIkwargs
        Several standard keyword arguments. See :class:`~.VIkwargs` for details.

    Raises
    ------
    NoVariablesError
        If the module does not have Bayesian parameters of its own and a method
        requiring them is called.
    """

    forward: Callable[..., _tensor_list_t] = _forward_unimplemented
    _return_log_probs: bool = True
    # _has_sampling_responsibility is set to False right after __init__ completes for
    # each submodule and True for itself that way submodules automatically call forward
    # and the outermost module calls sampled_forward instead
    _has_sampling_responsibility: bool
    _log_probs: Dict[str, List[Tensor]]

    def __init__(
        self,
        variable_shapes: Optional[Dict[str, Optional[Tuple[int, ...]]]] = None,
        variational_distribution: _dist_any_t = MeanFieldNormal(),
        prior: _dist_any_t = MeanFieldNormal(),
        rescale_prior: bool = False,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        convert_overwrite: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        if not convert_overwrite:
            super().__init__()

        if variable_shapes is None:
            return
        random_variables = tuple(variable_shapes.keys())

        if isinstance(variational_distribution, (tuple, list)):
            assert (
                len(variational_distribution) == len(random_variables)
            ), "Provide either exactly one variational distribution or exactly one for each random variable"
            for dist in variational_distribution:
                if not dist.is_variational_distribution:
                    raise UnsupportedDistributionError(
                        f"{dist.__class__.__name__} does not support use as variational distribution."
                    )
        else:
            if not variational_distribution.is_variational_distribution:
                raise UnsupportedDistributionError(
                    f"{variational_distribution.__class__.__name__} does not support use as variational distribution."
                )
            variational_distribution = [
                deepcopy(variational_distribution) for _ in random_variables
            ]
        self.variational_distribution = dict(
            zip(random_variables, variational_distribution)
        )

        if isinstance(prior, (tuple, list)):
            assert (
                len(prior) == len(random_variables)
            ), "Provide either exactly one prior distribution or exactly one for each random variable"
            for dist in prior:
                if not dist.is_prior:
                    raise UnsupportedDistributionError(
                        f"{dist.__class__.__name__} does not support use as prior."
                    )
        else:
            if not prior.is_prior:
                raise UnsupportedDistributionError(
                    f"{prior.__class__.__name__} does not support use as prior."
                )
            prior = [deepcopy(prior) for _ in random_variables]
        self.prior = dict(zip(random_variables, prior))

        if rescale_prior:
            self._rescale_prior(variable_shapes)

        self._kaiming_init = kaiming_initialization
        self._prior_init = prior_initialization
        self._return_log_probs = return_log_probs

        for var in random_variables:
            shape = variable_shapes[var]
            if shape is None:
                self.variational_distribution[var] = None
                self.prior[var] = None
                continue
            var_dist = self.variational_distribution[var]
            for var_param in var_dist.distribution_parameters:
                parameter_name = self.variational_parameter_name(var, var_param)
                setattr(
                    self,
                    parameter_name,
                    Parameter(
                        var_dist.empty_parameter(var_param, shape, **factory_kwargs)
                    ),
                )
        self._log_probs = dict()
        for variable in random_variables:
            self._log_probs[variable] = []
        self.reset_variational_parameters()

    @property
    def random_variables(self) -> Optional[Tuple[str, ...]]:
        """Names of the modules random variables."""
        if "variational_distribution" not in self.__dict__:
            return None
        return tuple(self.variational_distribution.keys())

    def _rescale_prior(
        self, variable_shapes: Dict[str, Optional[Tuple[int, ...]]]
    ) -> None:
        """
        Rescale the prior parameters based on the layer width.

        Experimental option. Related to Kaiming rescaling.

        Parameters
        ----------
        variable_shapes: Dict[str, Optional[Tuple[int, ...]]]
            The dictionary of random variable names and shapes as passed to __init__.

        Returns
        -------
        None
        """
        fan_in = self._calculate_fan_in(variable_shapes)
        for prior in self.prior.values():
            prior.kaiming_rescale(fan_in)

    def reset_variational_parameters(self) -> None:
        """
        Reset or initialize the parameters of the Module.

        Raises
        ------
        NoVariablesError
            If the module does not have parameters of its own.
        """
        if self.random_variables is None:
            raise NoVariablesError(
                f"{self.__class__.__name__} has no random variables to reset"
            )

        fan_in = self._calculate_fan_in()
        for var, dist in self.variational_distribution.items():
            if dist is None:
                continue
            dist.reset_variational_parameters(self, var, fan_in, self._kaiming_init)
            if self._prior_init:
                self.prior[var].reset_parameters_to_prior(self, var)

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

        Raises
        ------
        NoVariablesError
            If the module does not have parameters of its own.
        """
        if self.random_variables is None:
            raise NoVariablesError(
                f"{self.__class__.__name__} has no variational parameters to get"
            )

        vardist = self.variational_distribution[variable]
        return [
            getattr(self, self.variational_parameter_name(variable, param))
            for param in vardist.distribution_parameters
        ]

    def get_log_probs(self, sample: Tensor, variable: str) -> Tensor:
        """
        Get prior and variational log prob of the sampled parameters.

        Accepts the sampled parameters as returned by the `self.sample_variable` method
        and calculates the total prior and variational log probability.

        Parameters
        ----------
        sample: Tensor
            Sampled parameter as returned by the `self.sample_variable` method.
        variable: str
            The name of the relevant variable.

        Returns
        -------
        Tensor
            A Tensor containing two values: the log probability of the sampled
            values, if they were drawn from the prior or variational distribution (in
            that order).

        Raises
        ------
        NoVariablesError
            If the module does not have parameters of its own.
        """
        if self.random_variables is None:
            raise NoVariablesError(f"{self.__class__.__name__} has no random variables")

        vardist = self.variational_distribution[variable]
        prior = self.prior[variable]

        variational_parameters = self.get_variational_parameters(variable)
        variational_log_prob = vardist.variational_log_prob(
            sample, *variational_parameters
        ).sum()

        prior_params = [
            getattr(self, self.variational_parameter_name(variable, param))
            for param in prior._required_parameters
        ]
        prior_log_prob = prior.prior_log_prob(sample, *prior_params).sum()
        return torch.stack([prior_log_prob, variational_log_prob])

    def sample_variable(self, variable: str) -> Optional[Tensor]:
        """
        Draw one sample from the variational distribution of one random variable.

        This also performs log prob tracking, if :attr:`~self.return_log_probs` is True.

        Parameters
        ----------
        variable: str
            The variable to sample.

        Returns
        -------
        Tensor
            The sampled variable.

        Raises
        ------
        NoVariablesError
            If the module does not have parameters of its own.
        """
        if self.random_variables is None:
            raise NoVariablesError(
                f"{self.__class__.__name__} has no random variables to sample"
            )
        if self.variational_distribution[variable] is None:
            return None

        variational_parameters = self.get_variational_parameters(variable)
        sample = self.variational_distribution[variable].sample(*variational_parameters)

        if self.return_log_probs:
            log_probs = self.get_log_probs(sample, variable)
            try:
                log_probs = get_unwrapped(log_probs)
            except RuntimeError:
                pass
            self._log_probs[variable].append(log_probs)
        return sample

    @staticmethod
    def _expand_to_samples(input_: Optional[Tensor], samples: int) -> Tensor:
        if input_ is None:
            input_ = torch.tensor(False)
        return input_.expand(samples, *input_.shape)

    def sampled_forward(
        self, *input_: Optional[Tensor], samples: int = 10, **kwargs: Any
    ) -> Union[VIReturn, Tuple[VIReturn, ...]]:
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
        Union[VIReturn, Tuple[VIReturn, ...]]
            One or multiple Tensors with log prob annotation
        """
        expanded = [self._expand_to_samples(x, samples=samples) for x in input_]
        out: _tensor_list_t = torch.vmap(self.forward, randomness="different")(
            *expanded, **kwargs
        )

        if self.return_log_probs:
            log_probs = self.gather_log_probs()
        else:
            log_probs = None

        if isinstance(out, Tensor):
            return VIReturn.from_tensor(out, log_probs)

        for t in out:
            VIReturn.from_tensor(t, log_probs)
        return out

    def gather_log_probs(self) -> Tensor:
        """
        Gather and aggregate log probs from all submodules, then reset them.

        Returns
        -------
        Tensor
            The aggregated log probabilities of all submodules.
        """
        all_log_probs = []
        for module in self.modules():
            if not hasattr(module, "_log_probs"):
                continue
            for var, lps in module._log_probs.items():
                if len(lps) == 0:
                    pass
                else:
                    all_log_probs.append(torch.stack(lps).mean(dim=0))

                # Reset _log_probs
                module._log_probs[var] = []

        return torch.stack(all_log_probs).sum(dim=0)

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

    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        """Provide sampled weights under the common name."""
        if self.random_variables is not None and name in self.random_variables:
            return self.sample_variable(name)

        return super().__getattr__(name)

    def _calculate_fan_in(
        self, variable_shapes: Optional[Dict[str, Optional[Tuple[int, ...]]]] = None
    ) -> int:
        if variable_shapes is None:
            loop = self.random_variables
            checked_dict = self.variational_distribution
        else:
            loop = checked_dict = variable_shapes  # type:ignore[assignment]

        for var in cast(Tuple[str, ...], loop):
            if checked_dict[var] is None:
                continue

            if variable_shapes is None:
                weight_name = self.variational_parameter_name(
                    var,
                    self.variational_distribution[var].primary_parameter,
                )
                shape_dummy = getattr(self, weight_name)
            else:
                shape_dummy = torch.zeros(variable_shapes[var])
            fan_in, _ = init._calculate_fan_in_and_fan_out(shape_dummy)
            return fan_in

        raise NoVariablesError("All module variables are set to None.")
