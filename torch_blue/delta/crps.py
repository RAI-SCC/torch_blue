import math

import torch


def crps_gaussian_loss(
    mu_pred: torch.Tensor,
    var_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""
    Gaussian CRPS loss, averaged over all elements.

    Parameters
    ----------
    mu_pred: Tensor
        Predicted mean.
    var_pred: Tensor
        Predicted variance, clamped to ``[eps, 1e6]``.
    y_true: Tensor
        Observed target.
    eps: float, default: 1e-6
        Lower bound on the variance.

    Returns
    -------
    Tensor
        Scalar loss.

    """
    std_pred = torch.sqrt(torch.clamp(var_pred, min=eps, max=1e6))
    z = (y_true - mu_pred) / std_pred
    pdf_z = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf_z = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = std_pred * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / math.sqrt(math.pi))

    return crps.mean()
