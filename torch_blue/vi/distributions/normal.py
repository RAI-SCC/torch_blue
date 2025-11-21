from math import exp, log, prod
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions.utils import lazy_property
from torch.nn import init

from torch_blue.vi import _globals
from torch_blue.vi.utils.init import fixed_

from .base import Distribution

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


class MeanFieldNormal(Distribution):
    """
    Distribution assuming uncorrelated, normal distributed values.

    This distribution is implemented as prior, variational distribution, and predictive
    distribution. Its distribution parameters are "mean" and "log_std".

    As prior it becomes equivalent to an L2-weight decay term int the Kullback-Leibler
    loss.

    As variational distribution it is often the default assumption.

    As predictive distribution makes the Kullback-Leibler loss similar to MSE loss.

    Parameters
    ----------
    mean: float, default: 0.0
        The mean of the normal distribution before potential rescaling. Ignored if used
        as predictive distribution.
    std: float, default: 1.0
        The standard deviation of the normal distribution before potential rescaling.
        This is converted to a log std internally. Ignored if used as predictive
        distribution.
    eps: float, default: 1e-10
        Epsilon for numerical stability. Only relevant if used as prior.
    """

    is_prior: bool = True
    is_variational_distribution: bool = True
    is_predictive_distribution: bool = True

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-10) -> None:
        super().__init__()
        self.distribution_parameters = ("mean", "log_std")
        self.mean = mean
        self.log_std = log(std)
        self.eps = eps
        self._default_variational_parameters = (0.0, log(std))

    @lazy_property
    def std(self) -> float:
        """Prior standard deviation."""
        return exp(self.log_std)

    def sample(self, mean: Tensor, log_std: Tensor) -> Tensor:
        """
        Sample from a Gaussian distribution.

        Parameters
        ----------
        mean: Tensor
            The mean for each sample as Tensor.
        log_std: Tensor
            The log standard deviation for each sample as Tensor. Must have the same
            shape as `mean`.

        Returns
        -------
        Tensor
            The sampled Tensor of the same shape as `mean`.
        """
        std = torch.exp(log_std)
        return self._normal_sample(mean, std)

    @staticmethod
    def variational_log_prob(sample: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        """
        Compute the log probability of `sample` based on a normal distribution.

        Calculates the log probability of `sample` based on the provided mean and log
        standard deviation. All Tensors must have the same shape as `sample`.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            The weight configuration to calculate the log probability for.
        mean: Tensor
            The means of the reference distribution.
        log_std: Tensor
            The log standard deviations of the reference distribution.

        Returns
        -------
        Tensor
            The log probability of `sample` based on the provided mean and log_std.
        """
        variance = torch.exp(log_std) ** 2
        data_fitting = (sample - mean) ** 2 / variance
        normalization = 2 * log_std
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    @staticmethod
    def _normal_sample(mean: Tensor, std: Tensor) -> Tensor:
        base_sample = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        sample = std * base_sample + mean
        return sample

    def prior_log_prob(self, sample: Tensor) -> Tensor:
        """
        Compute the Gaussian log probability of a sample using the prior parameters.

        All Tensors have the same shape.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log probability for.

        Returns
        -------
        Tensor
            The log probability of the sample under the prior.
        """
        variance = self.std**2 + self.eps
        data_fitting = (sample - self.mean) ** 2 / variance
        normalization = 2 * self.log_std
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    def reset_parameters_to_prior(self, module: "VIModule", variable: str) -> None:
        """
        Reset the parameters of a module to prior mean and standard deviation.

        Parameters
        ----------
        module: VIModule
            The module containing the parameters to reset.
        variable: str
            The name of the random variable to reset as given by
            :attr:`variational_parameters` of the associated
            :class:`~torch_blue.vi.distributions.Distribution`.

        Returns
        -------
        None
        """
        mean_name = module.variational_parameter_name(variable, "mean")
        init.constant_(getattr(module, mean_name), self.mean)
        log_std_name = module.variational_parameter_name(variable, "log_std")
        init.constant_(getattr(module, log_std_name), self.log_std)

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Calculate predictive mean and standard deviation of samples.

        Parameters
        ----------
        samples: Tensor
            The model output as Tensor of shape (S, \*), where S is the number of
            samples.

        Returns
        -------
        Tensor
            The predictive mean as Tensor of shape (\*), i.e., the average along the
            sample dimension.
        Tensor
            The predictive standard deviation as Tensor of shape (\*), i.e., the
            standard deviation along the sample dimension.
        """
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """
        Calculate the log probability of reference given the predictive mean and standard deviation.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        reference: Tensor
            The ground truth label as Tensor of the same shape as each Tensor in
            `parameters`.
        parameters: Tuple[Tensor, Tensor]
            A tuple containing the predictive means and standard deviation as two
            Tensors as returned by :meth:`~predictive_parameters_from_samples`.

        Returns
        -------
        Tensor
            The log probability of the reference under the predicted normal distribution.
            Shape: (1,).
        """
        mean, std = parameters
        variance = std**2
        data_fitting = (reference - mean) ** 2 / variance
        normalization = torch.log(variance)
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)


class CorrelatedNormal(Distribution):
    """
    Distribution assuming correlated, normal distributed values.

    This distribution is implemented as prior, variational distribution, and predictive
    distribution. Its distribution parameters are "mean", "log_diag", and
    "cholesky_corr".

    Parameters
    ----------
    mean: float, default: 0.0
        The mean of the normal distribution before potential rescaling. Ignored if used
        as predictive distribution.
    std: float, default: 1.0
        The standard deviation of the normal distribution before potential rescaling.
        This is converted to a log std internally. For more fine-grained control,
        specify ``scale_tril``. Ignored if used as predictive distribution or if
        ``scale_tril`` is provided.
    corr: float, default: 0.0
        The correlations of the normal distribution before potential rescaling. For more
        fine-grained control, specify ``scale_tril``. Ignored if used as predictive
        distribution or if ``scale_tril`` is provided.
    scale_tril: Optional[Tensor], default: None
        If this is provided ``std`` and ``corr`` are ignored in favor of using this as
        the Cholesky decomposition of the covariance matrix.
    eps: float, default: 1e-10
        Epsilon for numerical stability. Only relevant if used as prior.
    """

    # is_prior: bool = True
    is_variational_distribution: bool = True
    # is_predictive_distribution: bool = True

    def __init__(
        self,
        mean: float,
        std: float = 1.0,
        corr: float = 0.0,
        covariance_matrix: Optional[Tensor] = None,
        scale_tril: Optional[Tensor] = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.distribution_parameters = ("mean", "log_diag", "cholesky_corr")
        self.mean = mean
        self.log_std = log(std)
        self.corr = corr
        self.eps = eps
        self._default_variational_parameters = (0.0, log(std), corr)

        if (covariance_matrix is not None) + (scale_tril is not None) > 1:
            raise ValueError(
                "Only one of covariance_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() != 2:
                raise ValueError("scale_tril must have two dimensions.")
            self.scale_tril = scale_tril
            self.covariance_matrix = torch.matmul(self.scale_tril, self.scale_tril.mT)
        if covariance_matrix is not None:
            if covariance_matrix.dim() != 2:
                raise ValueError("covariance_matrix must have two dimensions.")
            self.covariance_matrix = covariance_matrix
            self.scale_tril = torch.linalg.cholesky(covariance_matrix)

    @staticmethod
    def initialize_log_diag(
        variable_shape: Tuple[int, ...],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> Tensor:
        """
        Initialize an empty Tensor of the correct shape for log_diagonal.

        log_diag contains the log of diagonal elements of the Cholesky decomposition
        of the covariance matrix. Therefore, it is represented by a vector with one
        entry for each weight.

        Parameters
        ----------
        variable_shape: Tuple[int, ...]
            The shape of the associated weight matrix.
        device: Optional[torch.device]
            The target device.
        dtype: Optional[torch.dtype]
            The target data type.
        """
        return torch.empty(prod(variable_shape), device=device, dtype=dtype)

    @staticmethod
    def initialize_cholesky_corr(
        variable_shape: Tuple[int, ...],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> Tensor:
        """
        Initialize an empty Tensor of the correct shape for cholesky_corr.

        log_diag contains the non-diagonal elements of the Cholesky decomposition
        of the covariance matrix. Therefore, it is represented by a vector with
        N(N-1)/2 entries, where N is the number of weights.

        Parameters
        ----------
        variable_shape: Tuple[int, ...]
            The shape of the associated weight matrix.
        device: Optional[torch.device]
            The target device.
        dtype: Optional[torch.dtype]
            The target data type.
        """
        n = prod(variable_shape)
        return torch.empty(n * (n - 1) // 2, device=device, dtype=dtype)

    def sample(self, mean: Tensor, log_diag: Tensor, corr: Tensor) -> Tensor:
        """
        Sample from a correlated Gaussian distribution.

        Parameters
        ----------
        mean: Tensor
            The mean for each sample component as Tensor. Shape: (*,)
        log_diag: Tensor
            The log of the diagonal of the Cholesky decomposition of the covariance
            matrix. Shape: (N,), where N is the product of all elements in *.
        corr: Tensor
            The off-diagonal elements of the Cholesky decomposition of the covariance
            matrix. Shape: (N(N - 1)/2,)

        Returns
        -------
        Tensor
            Sampled Tensor of the same shape as ``mean``.
        """
        cholesky_tril = self._cholesky_from_params(corr, log_diag)
        base_sample = torch.normal(
            torch.zeros_like(log_diag), torch.ones_like(log_diag)
        )
        sample = torch.matmul(cholesky_tril, base_sample).reshape(mean.shape) + mean
        return sample

    @staticmethod
    def _cholesky_from_params(corr: Tensor, log_diag: Tensor) -> Tensor:
        len_diag = log_diag.shape[-1]
        cholesky_tril = torch.diag_embed(log_diag.exp())
        cholesky_tril[tuple(torch.tril_indices(len_diag, len_diag, offset=-1))] = corr
        return cholesky_tril

    def variational_log_prob(
        self,
        sample: Tensor,
        mean: Tensor,
        log_diag: Tensor,
        corr: Tensor,
    ) -> Tensor:
        """
        Compute the log probability of `sample` based on a normal distribution.

        Calculates the log probability of `sample` based on the provided mean and log
        standard deviation. All Tensors must have the same shape as `sample`.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            The weight configuration to calculate the log probability for.
        mean: Tensor
            The means of the reference distribution. Shape: (*,)
        log_diag: Tensor
            The log of the diagonal of the Cholesky decomposition of the covariance
            matrix. Shape: (N,), where N is the product of all elements in *.
        corr: Tensor
            The off-diagonal elements of the Cholesky decomposition of the covariance
            matrix. Shape: (N(N - 1)/2,)

        Returns
        -------
        Tensor
            The log probability of `sample` based on the provided parameters.
        """
        shifted_sample = (sample - mean).flatten()
        cholesky_tril = self._cholesky_from_params(corr, log_diag)
        inverse_covariance = torch.cholesky_inverse(cholesky_tril)

        data_fitting = torch.matmul(
            shifted_sample, torch.matmul(inverse_covariance, shifted_sample)
        )
        normalization = 2 * log_diag.sum()
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + len(log_diag) * log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    def prior_log_prob(self, sample: Tensor) -> Tensor:
        """
        Compute the correlated Gaussian log prob of a sample using the prior parameters.

        This calculation is affected by :data:`_globals._USE_NORM_CONSTANTS`, which can
        be set with :func:`~torch_blue.vi.utils.use_norm_constants`.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log probability for.

        Returns
        -------
        Tensor
            The log probability of the sample under the prior.
        """
        if self.scale_tril is not None:
            shifted_sample = (sample - self.mean).flatten()
            inverse_covariance = torch.cholesky_inverse(self.scale_tril)
            log_diag = self.scale_tril.diagonal().log()

            data_fitting = torch.matmul(
                shifted_sample, torch.matmul(inverse_covariance, shifted_sample)
            )
            normalization = 2 * log_diag.sum()
            if _globals._USE_NORM_CONSTANTS:
                normalization = normalization + len(log_diag) * log(2 * torch.pi)
            return -0.5 * (data_fitting + normalization)

        if self.corr == 0.0:
            return MeanFieldNormal.prior_log_prob(self, sample)  # type: ignore[arg-type]

        mean = torch.full_like(sample, self.mean)
        log_diag = torch.full_like(sample, self.log_std).flatten()
        n = len(log_diag)
        corr = torch.full(
            [2 * (n - 1) // 2], self.corr, dtype=sample.dtype, device=sample.device
        )
        return self.variational_log_prob(sample, mean, log_diag, corr)

    def reset_parameters_to_prior(self, module: "VIModule", variable: str) -> None:
        """
        Reset the parameters of a module to prior mean and covariance.

        Parameters
        ----------
        module: VIModule
            The module containing the parameters to reset.
        variable: str
            The name of the random variable to reset as given by
            :attr:`variational_parameters` of the associated
            :class:`~torch_blue.vi.distributions.Distribution`.

        Returns
        -------
        None
        """
        mean_name = module.variational_parameter_name(variable, "mean")
        log_diag_name = module.variational_parameter_name(variable, "log_diag")
        corr_name = module.variational_parameter_name(variable, "cholesky_corr")
        init.constant_(getattr(module, mean_name), self.mean)
        if self.scale_tril is None:
            init.constant_(getattr(module, log_diag_name), self.log_std)
            init.constant_(getattr(module, corr_name), self.corr)
        else:
            fixed_(getattr(module, log_diag_name), self.scale_tril.diagonal().log())
            len_diag = self.scale_tril.size(-1)
            fixed_(
                getattr(module, corr_name),
                self.scale_tril[
                    tuple(torch.tril_indices(len_diag, len_diag, offset=-1))
                ],
            )
