from typing import Callable, Optional

import torch
from torch import nn


class LearnableDelta(nn.Module):
    r"""
    Propagate input variance through a model with a stochastic delta method.

    The returned output variance is differentiable and is meant to be used as the
    predictive variance in a loss such as Gaussian NLL or CRPS.

    Parameters
    ----------
    model: Optional[Callable], default: None
        Model used when ``None`` is passed to :meth:`~forward`.
    increment: float, default: 1e-2
        Relative finite-difference step in mode ``"forward"``.
    mode: {"forward", "jacobian"}, default: "forward"
        Central finite difference or exact Jacobian-vector product.
    num_directions: int, default: 2
        Number of Rademacher probes.

    """

    def __init__(
        self,
        model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        increment: float = 1e-2,
        mode: str = "forward",
        num_directions: int = 2,
    ):
        super().__init__()
        if mode not in ("forward", "jacobian"):
            raise ValueError(f"mode must be 'forward' or 'jacobian', got {mode!r}")
        if num_directions < 1:
            raise ValueError(f"num_directions must be >= 1, got {num_directions}")
        if increment <= 0:
            raise ValueError(f"increment must be > 0, got {increment}")
        self.model = model
        self.increment = increment
        self.mode = mode
        self.num_directions = num_directions

    def forward(
        self,
        model: Optional[Callable[[torch.Tensor], torch.Tensor]],
        x: torch.Tensor,
        x_error: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the model output and its propagated variance.

        Parameters
        ----------
        model: Optional[Callable]
            Model to evaluate. ``None`` uses the stored model.
        x: Tensor
            Model input.
        x_error: Tensor
            Input variance, same shape as `x`.

        Returns
        -------
        y_mu: Tensor
            Model output.
        y_var: Tensor
            Output variance, same shape as `y_mu`.
        """
        model = self.model if model is None else model
        if model is None:
            raise ValueError("no model given and none stored on the module")
        k = self.num_directions
        y_mu = model(x)
        x_std = x_error.clamp_min(0).sqrt()

        y_var = torch.zeros_like(y_mu)
        for _ in range(k):
            if self.mode == "jacobian":
                diff = self.jacobian(model, x, x_std)
                y_var = y_var + diff**2
            elif self.mode == "forward":
                diff = self.zeroth_order(model, x, x_std)
                y_var = y_var + diff**2
            else:
                assert "mode must be 'forward' or 'jacobian'"
        return y_mu, y_var / k

    def jacobian(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Exact Jacobian-vector product along one random probe.

        Parameters
        ----------
        model: Callable
            Model to differentiate, compatible with :mod:`torch.func`.
        x: Tensor
            Point at which the Jacobian is evaluated.
        x_std: Tensor
            Input standard deviation, same shape as `x`.

        Returns
        -------
        Tensor
            :math:`J (\sigma_x \odot r)` for a Rademacher vector :math:`r`, same shape
            as the model output.

        """
        rademacher_tensor = (
            torch.randint(0, 2, x_std.shape, device=x_std.device).to(x_std.dtype) * 2
            - 1
        )
        d = x_std * rademacher_tensor
        _, diff = torch.func.jvp(model, (x,), (d,))

        return diff

    def zeroth_order(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Central finite-difference Jacobian-vector product along one random probe.

        Parameters
        ----------
        model: Callable
            Model to evaluate.
        x: Tensor
            Point at which the Jacobian is approximated.
        x_std: Tensor
            Input standard deviation, same shape as `x`.

        Returns
        -------
        Tensor
            Approximation of :math:`J (\sigma_x \odot r)` for a Rademacher vector
            :math:`r`, same shape as the model output.

        """
        rademacher_tensor = torch.randint(0, 2, x_std.shape).float() * 2 - 1
        d = x_std * rademacher_tensor
        diff = (model(x + self.increment * d) - model(x - self.increment * d)) / (
            2 * self.increment
        )

        return diff
