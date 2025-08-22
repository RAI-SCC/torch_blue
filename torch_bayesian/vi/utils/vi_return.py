from typing import Any, Optional

from torch import Tensor


class VIReturn(Tensor):
    """Likely dysfunctional log_prob tracking Tensor."""

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

    def clone(self, *args: Any, **kwargs: Any) -> "VIReturn":
        """Cloning."""
        if self.log_probs is None:
            return VIReturn(super().clone(*args, **kwargs), None)
        return VIReturn(
            super().clone(*args, **kwargs), self.log_probs.clone(*args, **kwargs)
        )

    def to(self, *args: Any, **kwargs: Any) -> "VIReturn":
        """To copy."""
        if self.log_probs is None:
            new_obj = VIReturn([], None)
        else:
            new_obj = VIReturn([], self.log_probs.to(*args, **kwargs))
        temp = super().to(*args, **kwargs)
        new_obj.data = temp.data
        new_obj.requires_grad = temp.requires_grad
        return new_obj
