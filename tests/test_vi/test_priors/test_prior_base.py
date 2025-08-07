from math import log, sqrt

import pytest
import torch
from torch import Tensor

from torch_bayesian.vi.priors import Prior


def test_parameter_checking() -> None:
    """Test enforcement or required parameters for subclasses of Prior."""

    # distribution_parameters assertion
    class Test1(Prior):
        pass

    with pytest.raises(
        NotImplementedError, match=r"Subclasses must define distribution_parameters"
    ):
        Test1()

    # log_prob assertion
    class Test2(Prior):
        distribution_parameters = ("mean", "log_std")

    with pytest.raises(NotImplementedError, match=r"Subclasses must define log_prob"):
        Test2()

    # log_prob signature assertion
    class Test3(Prior):
        distribution_parameters = ("mean", "log_std")
        _required_parameters = ("mean",)

        def log_prob(self, x: Tensor) -> Tensor:
            pass

    with pytest.raises(
        AssertionError,
        match=r"log_prob must accept an argument for each required parameter plus the sample",
    ):
        Test3()

    class Test4(Prior):
        distribution_parameters = ("mean", "log_std")

        def log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
            pass

    with pytest.raises(
        AssertionError,
        match=r"log_prob must accept an argument for each required parameter plus the sample",
    ):
        Test4()

    # Test scaling parameter enforcement
    class Test5(Prior):
        distribution_parameters = ("mean", "log_std")
        _required_parameters = ("mean",)

        def log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
            pass

    with pytest.raises(
        AssertionError,
        match=r"Module \[Test5\] is missing exposed scaling parameter \[mean\]",
    ):
        Test5()

    class Test6(Prior):
        distribution_parameters = ("mean", "log_std")
        mean: float = 0.0
        log_std: float = 0.0

        def log_prob(self, x: Tensor) -> Tensor:
            pass

    test = Test6()

    with pytest.warns(
        UserWarning,
        match=r'Module \[Test6\] is missing the "reset_parameters" method*',
    ):
        test.reset_parameters(test, "mean")  # type: ignore [arg-type]

    class Test7(Prior):
        distribution_parameters = ("mean", "log_std")
        _required_parameters = ("mean",)
        _scaling_parameters = ("log_std",)

        def __init__(self) -> None:
            self.log_std = torch.tensor(0.0)

        def log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
            pass

    _ = Test7()


def test_match_parameters() -> None:
    """Test Prior.match_parameters()."""

    class Test(Prior):
        distribution_parameters = ("mean", "log_std")
        _required_parameters = ("mean",)
        _scaling_parameters = ()

        def log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
            pass

    test = Test()

    ref1 = ("mean",)
    shared, diff = test.match_parameters(ref1)
    assert shared == {"mean": 0}
    assert diff == {"log_std": 1}

    ref2 = ("log_std", "mean", "skew")
    shared, diff = test.match_parameters(ref2)
    assert shared == {"mean": 0, "log_std": 1}
    assert diff == {}


def test_kaiming_rescale() -> None:
    """Test Prior.kaiming_rescale."""
    ref = dict(
        mean=1.0,
        log_std=0.0,
        skew=0.3,
        ff=2,
    )

    class Test(Prior):
        distribution_parameters = ("mean", "log_std", "skew", "ff")
        _scaling_parameters = ("mean", "log_std", "skew")
        mean: float = ref["mean"]
        log_std: float = ref["log_std"]
        skew: float = ref["skew"]
        ff: float = ref["ff"]

        def log_prob(self, x: Tensor) -> Tensor:
            pass

    # Test vector rescale
    test = Test()
    fan_in = 4
    test.kaiming_rescale(fan_in)

    scale = 1 / sqrt(3 * fan_in)
    assert test.mean == ref["mean"] * scale
    assert test.log_std == ref["log_std"] + log(scale + 1e-5)
    assert test.skew == ref["skew"] * scale
    assert test.ff == ref["ff"]

    # Test second rescale does nothing
    with pytest.warns(UserWarning, match="Test has already been rescaled.*"):
        test.kaiming_rescale(1, 9)
    assert test.mean == ref["mean"] * scale
    assert test.log_std == ref["log_std"] + log(1 * scale + 1e-5)
    assert test.skew == ref["skew"] * scale
    assert test.ff == ref["ff"]

    # Test matrix rescale
    test1 = Test()
    fan_in = 8
    eps = 1e-8
    test1.kaiming_rescale(fan_in, eps=eps)

    scale = 1 / sqrt(3 * fan_in)
    assert test1.mean == ref["mean"] * scale
    assert test1.log_std == ref["log_std"] + log(scale + eps)
    assert test1.skew == ref["skew"] * scale
    assert test1.ff == ref["ff"]

    # Test 0 fan_in
    test = Test()
    fan_in = 0
    eps = 1e-5
    test.kaiming_rescale(fan_in, eps=eps)

    assert test.mean == 0.0
    assert test.log_std == ref["log_std"] + log(eps)
    assert test.skew == 0.0
    assert test.ff == ref["ff"]
