from typing import Any, Optional

from torch import Tensor


class VIReturn(Tensor):
    """
    A subclass of :class:`torch.Tensor` that also stores log probabilities.

    A :class:`VIReturn` object behaves like a :class:`torch.Tensor` for all practical
    purposes, but provides the optional attribute :attr:`~log_probs`. However, it should
    not be used as replacement since certain pytorch operation will lose the log prob
    information. It is almost exclusively used as the return formate for
    :class:`~torch_blue.vi.VIModule`s. This allows the output to be treated like a
    :class:`torch.Tensor`, but still provide the log prob information when needed.
    `torch_blue` losses may require this format as input.
    """

    log_probs: Optional[Tensor] = None

    @staticmethod
    def __new__(
        cls, data: Tensor, log_probs: Optional[Tensor], *args: Any, **kwargs: Any
    ) -> "VIReturn":
        """Create a new VIReturn instance."""
        return super().__new__(cls, data, *args, **kwargs)

    def __init__(self, data: Tensor, log_probs: Optional[Tensor]) -> None:
        super().__init__()
        self.log_probs = log_probs

    # The doc strings of this and the next method throw warnings. This is inherited
    # from pytorch, but seems to compile correctly.
    def clone(self, *args: Any, **kwargs: Any) -> "VIReturn":  # noqa: D102
        if self.log_probs is None:
            return VIReturn(super().clone(*args, **kwargs), None)
        return VIReturn(
            super().clone(*args, **kwargs), self.log_probs.clone(*args, **kwargs)
        )

    def to(self, *args: Any, **kwargs: Any) -> "VIReturn":  # noqa: D102
        if self.log_probs is None:
            new_obj = VIReturn([], None)
        else:
            new_obj = VIReturn([], self.log_probs.to(*args, **kwargs))
        temp = super().to(*args, **kwargs)
        new_obj.data = temp.data
        new_obj.requires_grad = temp.requires_grad
        return new_obj

    @staticmethod
    def from_tensor(input_: Tensor, log_probs: Optional[Tensor]) -> "VIReturn":
        r"""
        Turn a torch.Tensor into a VIReturn.

        This is an inplace operation and does not copy data.

        Parameters
        ----------
        input\_: Tensor
            The Tensor to convert.
        log_probs: Optional[Tensor]
            The log probabilities to attach to input\_.

        Returns
        -------
        VIReturn
            The input converted to a VIReturn with the specified log probabilities.
        """
        input_.__class__ = VIReturn
        input_.log_probs = log_probs
        return input_
