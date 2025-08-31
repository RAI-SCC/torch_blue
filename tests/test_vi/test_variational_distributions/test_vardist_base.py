from math import log, sqrt

import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter

from torch_bayesian.vi.variational_distributions import VariationalDistribution


def test_parameter_checking() -> None:
    """Test enforcement or required parameters for subclasses of VariationalDistribution."""

    # variational_parameters assertion
    class Test(VariationalDistribution):
        pass

    try:
        Test()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define variational_parameters"

    # _default_variational_parameters assertion
    class Test1(VariationalDistribution):
        variational_parameters = ("mean", "std")

    try:
        Test1()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define _default_variational_parameters"

    # length matching of variational_parameters and default parameters
    class Test2(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0,)

    try:
        Test2()
        raise AssertionError
    except AssertionError as e:
        assert str(e) == "Each variational parameter must be assigned a default value"

    # sample assertion
    class Test3(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

    try:
        Test3()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define the sample method"

    # sample assertion
    class Test4(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor) -> Tensor:
            return mean

    try:
        Test4()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "Sample must accept exactly one Tensor for each variational parameter"
        )

    class Test5(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

    try:
        Test5()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define log_prob"

    class Test6(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def log_prob(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

    try:
        Test6()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "log_prob must accept an argument for each variational parameter plus the sample"
        )

    class Test7(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            return sample + mean + std

    _ = Test7()


def test_match_parameters() -> None:
    """Test VariationalDistribution.match_parameters()."""

    class Test(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
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
    VariationalDistribution._init_uniform(module.weight, fan_in)
    VariationalDistribution._init_uniform(module.bias, fan_in)
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
    VariationalDistribution._init_uniform(module.weight, fan_in)
    VariationalDistribution._init_uniform(module.bias, fan_in)
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
    VariationalDistribution._init_constant(module.weight, default[0], fan_in, False)
    VariationalDistribution._init_constant(module.bias, default[1], fan_in, False)
    weight2 = iter1.__next__().clone()
    bias2 = iter1.__next__().clone()

    assert (weight2 == (default[0] / sqrt(fan_in))).all()
    assert (bias2 == (default[1] / sqrt(fan_in))).all()
    assert weight2.device == device
    assert bias2.device == device

    iter2 = module.parameters()
    eps1 = 1e-5
    eps2 = 1e-3
    VariationalDistribution._init_constant(
        module.weight, default[0], fan_in, True, eps1
    )
    VariationalDistribution._init_constant(module.bias, default[1], 0, True, eps2)

    weight3 = iter2.__next__().clone()
    bias3 = iter2.__next__().clone()

    assert (weight3 == (default[0] + log(1 / sqrt(fan_in) + eps1))).all()
    assert (bias3 == default[1] + log(eps2)).all()
    assert weight3.device == device
    assert bias3.device == device


def test_vardist_reset_variational_parameters(device: torch.device) -> None:
    """Test VariationalDistribution.reset_variational_parameters."""
    param_shape = (5, 4)

    class Test(VariationalDistribution):
        variational_parameters = ("mean", "std", "log_std")
        _default_variational_parameters = (0.0, 1.0, 0.0)

        def sample(self, mean: Tensor, std: Tensor, log_std: Tensor) -> Tensor:
            return mean + std

        def log_prob(
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
