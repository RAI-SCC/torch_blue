import math
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, Tuple, Union
from warnings import warn

from torch import Tensor
from torch.nn import init

from ..utils import PostInitCallMeta

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


class Distribution(metaclass=PostInitCallMeta):
    r"""
    Base class for distributions.

    Distributions for variational implemented as subclasses of this. However, not all
    distributions can function or make sense in all available roles. Furthermore,
    each role has distinct requirements, which can be implemented independently.
    Available roles are prior, variational distribution, and predictive distribution.

    To simplify detection of correct usage this class has the three attributes
    :attr:`~self.is_prior`, :attr:`~self.is_variational_distribution`, and
    :attr:`~self.is_predictive_distribution`, which are `False` by default. When
    implementing custom distributions the flags for the intended usages must be set to
    `True`. This also implements initialization time checking that provides feedback if
    any required components for the flagged uses is missing.

    A prior specifies knowledge about the parameter distribution before training. In
    Bayesian training, weights are generally drawn towards the prior, unless they take
    an important role. Mathematically, this prior pull can take the same role as weight
    decay.


    Each prior must name the :attr:`~self.distribution_parameters` that define it as
    well as the way to calculate the log likelihood of a weight configuration in the
    :meth:`~self.prior_log_prob` method.

    Generally, each name in :attr:`~self.distribution_parameters` should also be an
    attribute of the class storing that parameter. This is necessary since the prior
    typically needs to be rescaled based on the layer width. By default, each parameter
    is assumed to require scaling, but in certain cases like the shape parameter of a
    Gamma distribution, this might not be the case. It that case the subset of scaling
    parameters must be specified in :attr:`~self._scaling_parameters`. Non-scaling
    parameters are technically not required to be a class attribute.

    Furthermore, parameters might only assume positive values. These should be stored as
    logarithm of their true value, mapping them to the whole real line. Their parameter
    name should begin with the prefix "log\_", which is automatically detected and
    handled during rescaling.

    To enable the ``prior_initialization`` functionality, the class must implement the
    :meth:`~self.reset_parameters` method. Which initializes the parameters for one
    random variable of a model, whose variational parameters are supported by the prior,
    to the prior values.

    Parameters
    ----------
    distribution_parameters: Tuple[str, ...]
        Parameters characterizing the prior and can be set during prior based
        initialization.
    _required_parameters: Tuple[str, ...], default: ()
        Parameters besides a sample needed to calculate :meth:`~log_prob`.
    _scaling_parameters: Tuple[str, ...], default: :attr:`~distribution_parameters`
        Parameters that need to be rescaled for prior rescaling.

    Methods
    -------
    prior_log_prob: Callable[..., Tensor]
        Function to calculate the log likelihood of a weight configuration under this
        prior.


    A variational distribution specifies the parametrization used to fit the true weight
    distribution, i.e., the weight posterior.

    Each prior must name the :attr:`~self.distribution_parameters` that will be
    optimized during training it as well as a default for each parameter in
    :attr:`~self._default_variational_parameters`. It is important to note that the
    first parameter is assumed to be a form of mean or mode of the distribution that
    might be used as the weight in a non-Bayesian version of the network. While a
    default value for it must be given, it will usually be ignored in initialization in
    favor of initializing it similar to non-Bayesian weights.

    Additionally, a :meth:`~self.sample` method must be defined that accepts one tensor
    for each variational parameter and returns a sample from the specified
    distributions. Finally, the way to calculate the log likelihood of a weight
    configuration in the :meth:`~self.variational_log_prob` method is required.

    Attributes
    ----------
    distribution_parameters : Tuple[str, ...]
        The names of the variational parameters that characterize the distribution.
        These are fit during training.
    _default_variational_parameters : Tuple[float, ...]
        Default initialization values for the variational parameters. If the parameter
        is "mean", "mode" or "loc", it is initialized analogously to non-Bayesian
        weights and the default is ignored.

    Methods
    -------
    sample : Callable[[Tensor, ...], Tensor]
        This method is used to sample the weigh matrices in the forward pass.
        The sample method needs to be implemented by each subclass. It accepts one
        Tensor for each variational parameter in the order specified in
        :attr:`variational_parameters` and returns a sample from the distribution
        of the same shape. All input Tensors must have the same shape.
    variational_log_prob : Callable[[Tensor, ...], Tensor]
        This method is used to calculate the log likelihood of a weight configuration
        under this distribution. The log_prob method needs to be implemented by each
        subclass. It accepts one Tensor containing the weight configuration and one
        Tensor for each variational parameter in the order specified in
        :attr:`variational_parameters` and returns the log likelihood of the weight
        configuration. All input Tensors must have the same shape.


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
    distribution_parameters: Tuple[str, ...]
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
            :attr:`~self.distribution_parameters`.

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
            in :attr:`~self.distribution_parameters`.

    Returns
    -------
        Tensor
            The log likelihood of the reference under the predicted distribution.
            Shape: (1,).
    """

    distribution_parameters: Tuple[str, ...]
    _default_variational_parameters: Tuple[float, ...]
    _required_parameters: Tuple[str, ...] = ()
    _scaling_parameters: Tuple[str, ...]
    _rescaled: bool = False
    sample: Callable[..., Tensor]
    prior_log_prob: Callable[..., Tensor]
    variational_log_prob: Callable[..., Tensor]
    predictive_parameters_from_samples: Callable[
        [Tensor], Union[Tensor, Tuple[Tensor, ...]]
    ]
    log_prob_from_parameters: Callable[
        [Tensor, Union[Tensor, Tuple[Tensor, ...]]], Tensor
    ]
    is_prior: bool = False
    is_variational_distribution: bool = False
    is_predictive_distribution: bool = False

    def reset_variational_parameters(
        self,
        module: "VIModule",
        variable: str,
        fan_in: int,
        kaiming_scaling: bool = True,
    ) -> None:
        """
        Reset the variational parameters of module.

        Parameters equivalent to non-Bayesian weights (currently "mean", "mode", or
        "loc") are reset accordingly using Kaiming uniform initialization based on
        `fan_in` (cf. :meth:`torch.init._calculate_fan_in_and_fan_out`). Other
        parameters are initialized to the fixed values specified by class defaults,
        i.e., :attr:`_default_variational_parameters`.
        If `kaiming_scaling` is ``True`` , the defaults are scaled with
        `scale` * `default`. Any parameter beginning with "log" is assumed to be in log
        space and scaled with `default` + log(`scale`). The scale is 1 / sqrt(`fan_in`)
        for vectors and 1 / sqrt(3 * `fan_in`) for matrices.

        Parameters
        ----------
        module: VIModule
            Module to reset parameters.
        variable: str
            Name of the variable to reset.
        fan_in: int
            Size of the input parameter map.
        kaiming_scaling: bool, default: True
            Whether th scale all parameters according to input map size.
        """
        for parameter, default in zip(
            self.distribution_parameters, self._default_variational_parameters
        ):
            parameter_name = module.variational_parameter_name(variable, parameter)
            param = getattr(module, parameter_name)

            if parameter in ["mean", "mode", "loc"]:
                self._init_uniform(param, fan_in)
            elif not kaiming_scaling:
                init.constant_(param, default)
            else:
                is_log = parameter.startswith("log")
                self._init_constant(param, default, fan_in, is_log)

    def reset_parameters_to_prior(self, module: "VIModule", variable: str) -> None:
        """
        Initialize the parameters of a VIModule according to the prior distribution.

        To enable the ``prior_initialization`` functionality, the class must implement
        this method. It initializes the parameters for one random variable of a
        :class:`~torchbuq.vi.VIModule`, whose variational parameters are
        supported by the prior, to the prior values. To that end the name of the random
        variable to initialize is passed to the method.

        This method is called separately for each submodule and therefore does not have
        to consider any further submodules. It is also called separately for each
        random variable but should manage all variational parameters the prior can
        provide. By convention, a prior whose distribution parameters are a true subset
        of the variational parameters initializes the parameters it can handle, e.g. a
        :class:`~.MeanFieldNormalPrior` can can be used to initialize parameters for any
        variational distribution that uses a `mean` and `log_std`.

        Parameters
        ----------
        module: VIModule
            The module containing the parameters to reset.
        variable: str
            The name of the random variable to reset as given by
            :attr:`variational_parameters` of the associated
            :class:`~torchbuq.vi.variational_distributions.VariationalDistribution`.

        Returns
        -------
        None
        """
        warn(
            f'Module [{type(self).__name__}] is missing the "reset_parameters_to_prior" method'
            f" and therefore does not support prior initialization."
        )

    @staticmethod
    def _init_constant(
        parameter: Tensor, default: float, fan_in: int, is_log: bool, eps: float = 1e-5
    ) -> None:
        scale = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        if is_log:
            init.constant_(parameter, default + math.log(scale + eps))
        else:
            init.constant_(parameter, scale * default)

    @staticmethod
    def _init_uniform(parameter: Tensor, fan_in: int) -> None:
        if parameter.dim() < 2:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(parameter, -bound, bound)
        else:
            init.kaiming_uniform_(parameter, a=math.sqrt(5))

    def __post_init__(self) -> None:
        """Ensure distribution is set up for specified modes."""
        if not (
            self.is_prior
            or self.is_variational_distribution
            or self.is_predictive_distribution
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__} not flagged to be functional as any distribution."
            )
        if self.is_prior:
            self.__prior_post_init__()
        if self.is_variational_distribution:
            self.__var_dist_post_init__()
        if self.is_predictive_distribution:
            self.__pred_dist_post_init__()

    def __prior_post_init__(self) -> None:
        """Ensure instance has required attributes to operate as prior."""
        if not hasattr(self, "distribution_parameters"):
            raise NotImplementedError("Subclasses must define distribution_parameters")
        if not hasattr(self, "prior_log_prob"):
            raise NotImplementedError("Subclasses must define prior_log_prob")
        assert (
            len(self._required_parameters)
            == (len(signature(self.prior_log_prob).parameters) - 1)
        ), "prior_log_prob must accept an argument for each required parameter plus the sample"
        if not hasattr(self, "_scaling_parameters"):
            self._scaling_parameters = self.distribution_parameters
        for parameter in self._scaling_parameters:
            assert hasattr(
                self, parameter
            ), f"Module [{type(self).__name__}] is missing exposed scaling parameter [{parameter}]"

    def __var_dist_post_init__(self) -> None:
        """Ensure instance has required attributes to operate as variational distribution."""
        if not hasattr(self, "distribution_parameters"):
            raise NotImplementedError("Subclasses must define distribution_parameters")
        if not hasattr(self, "_default_variational_parameters"):
            raise NotImplementedError(
                "Subclasses must define _default_variational_parameters"
            )
        assert len(self.distribution_parameters) == len(
            self._default_variational_parameters
        ), "Each variational parameter must be assigned a default value"
        if not hasattr(self, "sample"):
            raise NotImplementedError("Subclasses must define the sample method")
        assert len(self.distribution_parameters) == len(
            signature(self.sample).parameters
        ), "Sample must accept exactly one Tensor for each distribution parameter"
        if not hasattr(self, "variational_log_prob"):
            raise NotImplementedError("Subclasses must define variational_log_prob")
        assert (
            len(self.distribution_parameters)
            == (len(signature(self.variational_log_prob).parameters) - 1)
        ), "variational_log_prob must accept an argument for each variational parameter plus the sample"

    def __pred_dist_post_init__(self) -> None:
        """Ensure instance has all required attributes to operate as predictive distribution."""
        if not hasattr(self, "distribution_parameters"):
            raise NotImplementedError("Subclasses must define distribution_parameters")
        if not hasattr(self, "predictive_parameters_from_samples"):
            raise NotImplementedError(
                "Subclasses must define predictive_parameters_from_samples"
            )
        if not hasattr(self, "log_prob_from_parameters"):
            raise NotImplementedError("Subclasses must define log_prob_from_parameters")

    def match_parameters(
        self, distribution_parameters: Tuple[str, ...]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Compare distribution parameters to another set of parameters.

        Typically, this is used to compare to the distribution parameters of a
        :class:`~torchbuq.vi.priors.Prior`.

        Parameters
        ----------
        distribution_parameters : Tuple[str]
            Tuple of parameter names to compare.

        Returns
        -------
        Tuple[Dict[str, int], Dict[str, int]]
            The first dictionary maps the names of the shared parameters to their index
            in :attr:`~self.variational_parameters`. The second dictionary maps the
            names of parameters exclusive to :attr:`~self.variational_parameters` to
            their index.
        """
        shared_params = {}
        diff_params = {}

        for i, var_param in enumerate(self.distribution_parameters):
            if var_param in distribution_parameters:
                shared_params[var_param] = i
            else:
                diff_params[var_param] = i

        return shared_params, diff_params

    def kaiming_rescale(self, fan_in: int, eps: float = 1e-5) -> None:
        r"""
        Rescale the prior based on layer width, for normalization.

        Parameters from :attr:`~self._scaling_parameters` are scaled linearly based on
        the square root of the layer width, unless their name begins with "log\_", in
        which case they are scaled such that their exponential scales in the same way.

        Parameters
        ----------
        fan_in : int
            The relevant layer width.
        eps: float, default: 1e-5
            Epsilon for numerical stability.

        Returns
        -------
        None
        """
        if self._rescaled:
            warn(
                f"{type(self).__name__} has already been rescaled. Ignoring rescaling."
            )
            pass
        else:
            self._rescaled = True
            scale = 1 / math.sqrt(3 * fan_in) if fan_in > 0 else 0

            for parameter in self._scaling_parameters:
                param = getattr(self, parameter)
                if parameter.startswith("log"):
                    setattr(self, parameter, param + math.log(scale + eps))
                else:
                    setattr(self, parameter, param * scale)

    @property
    def primary_parameter(self) -> str:
        """The distribution parameter that is closest to a non-Bayesian weight."""
        return self.distribution_parameters[0]

    def log_prob_from_samples(self, reference: Tensor, samples: Tensor) -> Tensor:
        r"""
        Calculate the log probability for reference given a set of samples.

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
            The log probability of the reference under the predicted distribution.
            Shape: (1,).
        """
        params = self.predictive_parameters_from_samples(samples)
        return self.log_prob_from_parameters(reference, params)
