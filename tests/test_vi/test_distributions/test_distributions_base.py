from math import log, sqrt
from typing import Dict, Tuple

import pytest
import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter

from torchbuq.vi.distributions import Distribution


class TestDistribution:
    """Tests for Distribution class."""

    target = Distribution

    def _simple_log_prob(self, x: Tensor) -> Tensor:
        pass

    def _required_mean_log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
        pass

    @pytest.mark.parametrize(
        "params",
        [
            {"foo": 0.0, "bar": 1.0},
            {"mean": 0.0, "log_std": 0.0},
            {"mean": 1.0},
            {"mean": 0.0, "log_std": 0.0, "df": 3.0},
        ],
    )
    def test_primary_parameters(self, params: Dict[str, float]) -> None:
        """Test primary parameter property."""

        class Test(self.target):  # type:ignore [name-defined]
            is_prior = True
            distribution_parameters = tuple(params.keys())
            prior_log_prob = self._simple_log_prob

        for name, val in params.items():
            setattr(Test, name, val)

        test = Test()
        assert test.primary_parameter == tuple(params.keys())[0]

    def test_parameter_checking_prior(self) -> None:
        """Test enforcement or required parameters for subclasses of Distribution, prior mode."""

        class Test(self.target):  # type:ignore [name-defined]
            pass

        with pytest.raises(
            NotImplementedError,
            match="Test not flagged to be functional as any distribution.",
        ):
            Test()

        # distribution_parameters assertion
        Test.is_prior = True

        with pytest.raises(
            NotImplementedError, match=r"Subclasses must define distribution_parameters"
        ):
            Test()

        # log_prob assertion
        Test.distribution_parameters = ("mean", "log_std")

        with pytest.raises(
            NotImplementedError, match=r"Subclasses must define prior_log_prob"
        ):
            Test()

        # log_prob signature without required parameters
        Test.prior_log_prob = self._required_mean_log_prob

        with pytest.raises(
            AssertionError,
            match=r"prior_log_prob must accept an argument for each required parameter plus the sample",
        ):
            Test()

        # log_prob signature with required parameters
        Test.prior_log_prob = self._simple_log_prob
        Test._required_parameters = ("mean",)

        with pytest.raises(
            AssertionError,
            match=r"log_prob must accept an argument for each required parameter plus the sample",
        ):
            Test()

        # Test scaling parameter enforcement
        Test._required_parameters = ()

        with pytest.raises(
            AssertionError,
            match=r"Module \[Test\] is missing exposed scaling parameter \[mean\]",
        ):
            Test()

        Test.mean = 0.0
        Test.log_std = 0.0

        test = Test()

        with pytest.warns(
            UserWarning,
            match=r'Module \[Test\] is missing the "reset_parameters_to_prior" method*',
        ):
            test.reset_parameters_to_prior(test, "mean")  # type: ignore [arg-type]

        Test._required_parameters = ("mean",)
        Test._scaling_parameters = ("log_std",)
        del Test.mean
        Test.prior_log_prob = self._required_mean_log_prob

        _ = Test()


def test_kaiming_rescale() -> None:
    """Test Prior.kaiming_rescale."""
    ref = dict(
        mean=1.0,
        log_std=0.0,
        skew=0.3,
        ff=2,
    )

    class Test(Distribution):
        is_prior = True
        distribution_parameters = ("mean", "log_std", "skew", "ff")
        _scaling_parameters = ("mean", "log_std", "skew")
        mean: float = ref["mean"]
        log_std: float = ref["log_std"]
        skew: float = ref["skew"]
        ff: float = ref["ff"]

        def prior_log_prob(self, x: Tensor) -> Tensor:
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


def test_parameter_checking_variational() -> None:
    """Test enforcement or required parameters for subclasses of Distribution, variational mode."""

    # variational_parameters assertion
    class Test(Distribution):
        is_variational_distribution = True

    try:
        Test()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define distribution_parameters"

    # _default_variational_parameters assertion
    class Test1(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")

    try:
        Test1()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define _default_variational_parameters"

    # length matching of variational_parameters and default parameters
    class Test2(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")
        _default_variational_parameters = (0.0,)

    try:
        Test2()
        raise AssertionError
    except AssertionError as e:
        assert str(e) == "Each variational parameter must be assigned a default value"

    # sample assertion
    class Test3(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

    try:
        Test3()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define the sample method"

    # sample assertion
    class Test4(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor) -> Tensor:
            return mean

    try:
        Test4()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "Sample must accept exactly one Tensor for each distribution parameter"
        )

    class Test5(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

    try:
        Test5()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define variational_log_prob"

    class Test6(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def variational_log_prob(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

    try:
        Test6()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "variational_log_prob must accept an argument for each variational parameter plus the sample"
        )

    class Test7(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def variational_log_prob(
            self, sample: Tensor, mean: Tensor, std: Tensor
        ) -> Tensor:
            return sample + mean + std

    _ = Test7()


def test_match_parameters() -> None:
    """Test Distribution.match_parameters()."""

    class Test(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def variational_log_prob(
            self, sample: Tensor, mean: Tensor, std: Tensor
        ) -> Tensor:
            return sample + mean + std

    test = Test()

    ref1 = ("mean",)
    shared, diff = test.match_parameters(ref1)
    assert shared == {"mean": 0}
    assert diff == {"std": 1}

    ref2 = ("std", "mean", "skew")
    shared, diff = test.match_parameters(ref2)
    assert shared == {"mean": 0, "std": 1}
    assert diff == {}


def test_init_uniform(device: torch.device) -> None:
    """Test _init_uniform."""
    parameter_shape = (6, 20)
    module = Linear(*parameter_shape, device=device)

    weight1 = module.weight.clone()
    bias1 = module.bias.clone()
    iter1 = module.parameters()
    fan_in = 10
    Distribution._init_uniform(module.weight, fan_in)
    Distribution._init_uniform(module.bias, fan_in)
    weight2 = iter1.__next__().clone()
    bias2 = iter1.__next__().clone()

    assert not torch.allclose(weight1, weight2)
    assert not torch.allclose(bias1, bias2)
    assert weight1.device == device
    assert weight2.device == device
    assert bias1.device == device
    assert bias2.device == device

    assert (weight2.abs() < (1 / sqrt(parameter_shape[0]))).all()
    assert (bias2.abs() < (1 / sqrt(fan_in))).all()

    iter2 = module.parameters()
    fan_in = 0
    Distribution._init_uniform(module.weight, fan_in)
    Distribution._init_uniform(module.bias, fan_in)
    weight3 = iter2.__next__().clone()
    bias3 = iter2.__next__().clone()

    assert not torch.allclose(weight2, weight3)
    assert (weight3.abs() < (1 / sqrt(parameter_shape[0]))).all()
    assert (bias3 == 0).all()
    assert weight3.device == device
    assert bias3.device == device


def test_init_constant(device: torch.device) -> None:
    """Test _init_constant."""
    parameter_shape = (5, 15)
    module = Linear(*parameter_shape, device=device)

    iter1 = module.parameters()
    default = (1.0, 2.0)
    fan_in = 7
    Distribution._init_constant(module.weight, default[0], fan_in, False)
    Distribution._init_constant(module.bias, default[1], fan_in, False)
    weight2 = iter1.__next__().clone()
    bias2 = iter1.__next__().clone()

    assert (weight2 == (default[0] / sqrt(fan_in))).all()
    assert (bias2 == (default[1] / sqrt(fan_in))).all()
    assert weight2.device == device
    assert bias2.device == device

    iter2 = module.parameters()
    eps1 = 1e-5
    eps2 = 1e-3
    Distribution._init_constant(module.weight, default[0], fan_in, True, eps1)
    Distribution._init_constant(module.bias, default[1], 0, True, eps2)

    weight3 = iter2.__next__().clone()
    bias3 = iter2.__next__().clone()

    assert (weight3 == (default[0] + log(1 / sqrt(fan_in) + eps1))).all()
    assert (bias3 == default[1] + log(eps2)).all()
    assert weight3.device == device
    assert bias3.device == device


def test_vardist_reset_variational_parameters(device: torch.device) -> None:
    """Test VariationalDistribution.reset_variational_parameters."""
    param_shape = (5, 4)

    class Test(Distribution):
        is_variational_distribution = True
        distribution_parameters = ("mean", "std", "log_std")
        _default_variational_parameters = (0.0, 1.0, 0.0)

        def sample(self, mean: Tensor, std: Tensor, log_std: Tensor) -> Tensor:
            return mean + std

        def variational_log_prob(
            self, sample: Tensor, mean: Tensor, std: Tensor, log_std: Tensor
        ) -> Tensor:
            return sample + mean + std

    class ModuleDummy(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_mean = Parameter(torch.empty(param_shape))
            self.weight_std = Parameter(torch.empty(param_shape))
            self.weight_log_std = Parameter(torch.empty(param_shape[0]))

        @staticmethod
        def variational_parameter_name(variable: str, parameter: str) -> str:
            return f"{variable}_{parameter}"

    vardist = Test()
    dummy = ModuleDummy().to(device=device)
    fan_in = param_shape[1]

    mean1 = dummy.weight_mean.clone()
    iter1 = dummy.parameters()

    vardist.reset_variational_parameters(dummy, "weight", fan_in, False)
    mean2 = iter1.__next__().clone()
    std2 = iter1.__next__().clone()
    log_std2 = iter1.__next__().clone()

    assert not torch.allclose(mean1, mean2)
    assert len(torch.unique(mean2)) == param_shape[0] * param_shape[1]
    assert mean1.device == device
    assert mean2.device == device
    assert std2.device == device

    assert std2.shape == param_shape
    assert (std2 == 1.0).all()
    assert log_std2.shape == param_shape[:1]
    assert (log_std2 == 0.0).all()

    iter2 = dummy.parameters()
    eps = 1e-5

    vardist.reset_variational_parameters(dummy, "weight", fan_in, True)
    mean3 = iter2.__next__().clone()
    std3 = iter2.__next__().clone()
    log_std3 = iter2.__next__().clone()

    assert not torch.allclose(mean2, mean3)
    assert len(torch.unique(mean3)) == param_shape[0] * param_shape[1]
    assert mean3.device == device
    assert std3.device == device

    assert std3.shape == param_shape
    assert (std3 == (1.0 / sqrt(fan_in))).all()
    assert log_std3.shape == param_shape[:1]
    assert (log_std3 == 0.0 + log(1 / sqrt(fan_in) + eps)).all()


def test_parameter_checking_predictive() -> None:
    """Test enforcement or required parameters for subclasses of Distribution, predictive mode."""

    # predictive_parameters assertion
    class Test1(Distribution):
        is_predictive_distribution = True

    try:
        Test1()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define distribution_parameters"

    # predictive_parameters_from_samples assertion
    class Test2(Distribution):
        is_predictive_distribution = True
        distribution_parameters = ("mean", "log_std")

    try:
        Test2()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define predictive_parameters_from_samples"

    # log_prob_from_parameters assertion
    class Test3(Distribution):
        is_predictive_distribution = True
        distribution_parameters = ("mean", "log_std")

        def predictive_parameters_from_samples(
            self, samples: Tensor
        ) -> Tuple[Tensor, Tensor]:
            return samples, samples

    try:
        Test3()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define log_prob_from_parameters"

    # Test correct init
    class Test4(Distribution):
        is_predictive_distribution = True
        distribution_parameters = ("mean", "log_std")

        def predictive_parameters_from_samples(
            self, samples: Tensor
        ) -> Tuple[Tensor, Tensor]:
            return samples, samples

        def log_prob_from_parameters(
            self, reference: Tensor, parameters: Tuple[Tensor, Tensor]
        ) -> Tensor:
            return reference

    _ = Test4()


def test_log_prob_from_samples(device: torch.device) -> None:
    """Test Distribution.log_prob_from_samples."""

    class Test(Distribution):
        is_predictive_distribution = True
        distribution_parameters = ("mean",)

        @staticmethod
        def predictive_parameters_from_samples(samples: Tensor) -> Tensor:
            return samples.sum(dim=0)

        @staticmethod
        def log_prob_from_parameters(reference: Tensor, parameters: Tensor) -> Tensor:
            return reference + parameters

    test = Test()
    samples = torch.randn((5, 3, 4), device=device)
    reference = torch.randn((3, 4), device=device)
    target = samples.sum(dim=0) + reference

    out = test.log_prob_from_samples(reference, samples)
    assert torch.allclose(target, out)
    assert out.device == device
