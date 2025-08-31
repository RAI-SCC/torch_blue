import math
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, Tuple
from warnings import warn

from torch import Tensor

from ..utils import PostInitCallMeta

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


class Prior(metaclass=PostInitCallMeta):
    r"""
    Base class for all prior distributions.

    A prior specifies knowledge about the parameter distribution before training. In
    Bayesian training, weights are generally drawn towards the prior, unless they take
    an important role. Mathematically, this prior pull can take the same role as weight
    decay.

    Each prior must name the :attr:`~self.distribution_parameters` that define it as
    well as the way to calculate the log probability of a weight configuration in the
    :meth:`~self.log_prob` method.

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
    :meth:`~self.reset_variational_parameters` method. Which initializes the parameters for one
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
    log_prob: Callable[..., Tensor]
        Function to calculate the log probability of a weight configuration under this
        prior.
    """

    distribution_parameters: Tuple[str, ...]
    _required_parameters: Tuple[str, ...] = ()
    _scaling_parameters: Tuple[str, ...]
    _rescaled: bool = False
    log_prob: Callable[..., Tensor]

    def __post_init__(self) -> None:
        """Ensure the instance has required attributes."""
        if not hasattr(self, "distribution_parameters"):
            raise NotImplementedError("Subclasses must define distribution_parameters")
        if not hasattr(self, "log_prob"):
            raise NotImplementedError("Subclasses must define log_prob")
        assert (
            len(self._required_parameters)
            == (len(signature(self.log_prob).parameters) - 1)
        ), "log_prob must accept an argument for each required parameter plus the sample"
        if not hasattr(self, "_scaling_parameters"):
            self._scaling_parameters = self.distribution_parameters
        for parameter in self._scaling_parameters:
            assert hasattr(
                self, parameter
            ), f"Module [{type(self).__name__}] is missing exposed scaling parameter [{parameter}]"

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

    def reset_variational_parameters(self, module: "VIModule", variable: str) -> None:
        """
        Initialize the parameters of a VIModule according to the prior distribution.

        To enable the ``prior_initialization`` functionality, the class must implement
        this method. It initializes the parameters for one random variable of a
        :class:`~torch_bayesian.vi.VIModule`, whose variational parameters are
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
            :class:`~torch_bayesian.vi.variational_distributions.VariationalDistribution`.

        Returns
        -------
        None
        """
        warn(
            f'Module [{type(self).__name__}] is missing the "reset_variational_parameters" method'
            f" and therefore does not support prior initialization."
        )

    def match_parameters(
        self, variational_parameters: Tuple[str, ...]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Compare distribution parameters to another set of parameters.

        Typically, this is used to compare to the variational parameters of a
        :class:`~torch_bayesian.vi.variational_distributions.VariationalDistribution`.

        Parameters
        ----------
        variational_parameters : Tuple[str]
            Tuple of parameter names to compare.

        Returns
        -------
        Tuple[Dict[str, int], Dict[str, int]]
            The first dictionary maps the names of the shared parameters to their index
            in :attr:`~self.distribution_parameters`. The second dictionary maps the
            names of parameters exclusive to :attr:`~self.distribution_parameters` to
            their index.
        """
        shared_params = {}
        diff_params = {}

        for i, dist_param in enumerate(self.distribution_parameters):
            if dist_param in variational_parameters:
                shared_params[dist_param] = i
            else:
                diff_params[dist_param] = i

        return shared_params, diff_params
