from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .base import VIBaseModule
from .priors import MeanFieldNormalPrior
from .utils.common_types import VIkwargs, VIReturn, _prior_any_t, _vardist_any_t
from .variational_distributions import MeanFieldNormalVarDist


class VILinear(VIBaseModule):
    """
    Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    Equivalent of :class:`nn.Linear` with variational inference. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`__
    for usage.

    In addition to those arguments, this class accepts :class:`~.VIkwargs`.

    This module's random variables are

    - ("weight", "bias") if bias == True
    - ("weight", )       if bias == False
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        bias: bool = True,
        rescale_prior: bool = False,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
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
        self.in_features = in_features
        self.out_features = out_features

        if bias:
            self.random_variables = ("weight", "bias")
        else:
            self.random_variables = ("weight",)

        variable_shapes = dict(
            weight=(out_features, in_features),
            bias=(out_features,),
        )

        super().__init__(variable_shapes=variable_shapes, **vikwargs)

        # If the variational distribution is stable we might be able to use the stable fast path
        if all(
            isinstance(dist, MeanFieldNormalVarDist)
            for dist in self.variational_distribution
        ):
            self._fast_path = True
        else:
            self._fast_path = False

    def forward(self, input_: Tensor) -> VIReturn[Tensor]:
        r"""
        Forward computation.

        Parameters
        ----------
        input_: Tensor
            Input tensor of shape (\*, in_features).

        Returns
        -------
        output: Tensor
            Output tensor of shape (\*, out_features). Auto-sampling will add a sample
            dimension at the start for the overall output.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of the sampled weights and biases.

            Only returned if ``return_log_probs``. Otherwise, only **output** is returned.
        """
        # Check for and perform fast path if possible:
        if (not self._return_log_probs) and self._fast_path:
            output = self._fast_forward(input_)
            return output

        params = self.sample_variables()

        output = F.linear(input_, *params)

        if self._return_log_probs:
            log_probs = self.get_log_probs(params)
            return output, log_probs
        else:
            return output

    def _fast_forward(self, input_: Tensor) -> Tensor:
        """Perform the stable fast path for Gaussian variational distribution."""
        weight_mean = self._weight_mean
        weight_variance = (2 * self._weight_log_std).exp()
        if "bias" in self.random_variables:
            bias_mean = self._bias_mean
            bias_variance = (2 * self._bias_log_std).exp()
        else:
            bias_mean = None
            bias_variance = None
        output_mean = F.linear(input_, weight_mean, bias_mean)
        output_std = F.linear(input_.pow(2), weight_variance, bias_variance).sqrt()
        output = MeanFieldNormalVarDist._normal_sample(output_mean, output_std)
        return output
