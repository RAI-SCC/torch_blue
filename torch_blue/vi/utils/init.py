import torch
from torch import Tensor


def fixed_(tensor: Tensor, other: Tensor) -> Tensor:
    """
    Initialize tensor to be equal to other.

    Initialize a Tensor to the values of another Tensor of the same shape.

    Parameters
    ----------
    tensor: Tensor
        The Tensor to be initialized.
    other: Tensor
        The Tensor whose values are used.

    Returns
    -------
    Tensor
        The initialized Tensor.
    """
    assert (
        tensor.size() == other.size()
    ), "Values must be provided for all tensor elements."
    with torch.no_grad():
        return tensor.copy_(other)
