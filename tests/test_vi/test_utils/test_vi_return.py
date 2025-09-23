from typing import Tuple, cast

import pytest
import torch
from torch import Tensor

from torch_bayesian.vi import VIReturn


class TestVIReturn:
    """Test the VIReturn class."""

    @staticmethod
    def _init_instance(
        shape: Tuple[int], lp_is_none: bool = False
    ) -> Tuple[VIReturn, Tensor, Tensor]:
        ref_tensor = torch.randn(shape)
        ref_log_probs = None if lp_is_none else torch.randn([10, 2])
        vi_return = VIReturn(ref_tensor, ref_log_probs)
        return vi_return, ref_tensor, ref_log_probs

    def test_init(self) -> None:
        """Test initialization."""
        shape = tuple(torch.randint(1, 10, [3]))
        vi_return, ref_tensor, ref_log_probs = self._init_instance(shape)
        assert isinstance(vi_return, torch.Tensor)
        assert torch.all(vi_return == ref_tensor)
        assert vi_return is not ref_tensor
        assert torch.all(vi_return.log_probs == ref_log_probs)
        assert vi_return.log_probs is ref_log_probs

    @pytest.mark.parametrize("lp_is_none", [True, False])
    def test_clone(self, lp_is_none: bool) -> None:
        """Test clone."""
        shape = tuple(torch.randint(1, 10, [3]))
        vi_return, ref_tensor, ref_log_probs = self._init_instance(shape, lp_is_none)
        new = vi_return.clone()
        assert new is not vi_return
        assert torch.all(ref_tensor == new)

        if lp_is_none:
            assert new.log_probs is None
        else:
            assert new.log_probs is not vi_return.log_probs
            assert torch.all(new.log_probs == ref_log_probs)

    @pytest.mark.parametrize(
        "dtype,lp_is_none",
        [
            (torch.float, True),
            (torch.float, False),
            (torch.double, True),
            (torch.double, False),
        ],
    )
    def test_to(self, dtype: torch.dtype, lp_is_none: bool) -> None:
        """Test to method."""
        shape = tuple(torch.randint(1, 10, [3]))
        vi_return, ref_tensor, ref_log_probs = self._init_instance(shape, lp_is_none)
        new = vi_return.to(dtype=dtype)
        assert new is not vi_return
        assert torch.all(ref_tensor == new)
        assert new.dtype is dtype

        if lp_is_none:
            assert new.log_probs is None
        else:
            assert torch.all(new.log_probs == ref_log_probs)
            assert cast(Tensor, new.log_probs).dtype is dtype

    @pytest.mark.parametrize("lp_is_none", [True, False])
    def test_from_tensor(self, lp_is_none: bool) -> None:
        """Test from_tensor."""
        shape = tuple(torch.randint(1, 10, [3]))
        vi_return, ref_tensor, ref_log_probs = self._init_instance(shape, lp_is_none)
        new = VIReturn.from_tensor(ref_tensor, ref_log_probs)
        assert new is not vi_return
        assert new is ref_tensor
        assert torch.all(ref_tensor == new)

        if lp_is_none:
            assert new.log_probs is None
        else:
            assert new.log_probs is vi_return.log_probs
            assert torch.all(new.log_probs == ref_log_probs)
