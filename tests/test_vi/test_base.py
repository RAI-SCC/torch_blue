import math
from typing import Any, List, Tuple, Union, cast
from warnings import filterwarnings

import pytest
import torch
from torch import Tensor
from torch._C._functorch import get_unwrapped
from torch.nn import Module

from torch_bayesian.vi import VIModule
from torch_bayesian.vi.priors import Prior
from torch_bayesian.vi.utils import NoVariablesError
from torch_bayesian.vi.variational_distributions import VariationalDistribution


def test_expand_to_samples(device: torch.device) -> None:
    """Test _expand_to_samples."""
    shape1 = (3, 4)
    sample1 = torch.randn(shape1, device=device)
    out1 = VIModule._expand_to_samples(sample1, samples=5)
    assert out1.shape == (5,) + shape1
    for s in out1:
        assert (s == sample1).all()

    shape2 = (5,)
    sample2 = torch.randn(shape2, device=device)
    out2 = VIModule._expand_to_samples(sample2, samples=1)
    assert out2.shape == (1,) + shape2
    for s in out2:
        assert (s == sample2).all()

    out3 = VIModule._expand_to_samples(None, samples=2)
    assert (out3 == torch.tensor([False, False])).all()


def test_no_forward_error() -> None:
    """Test that forward throws error if not implemented."""
    module = VIModule()
    with pytest.raises(
        NotImplementedError,
        match=r'Module \[VIModule\] is missing the required "forward" function',
    ):
        module.forward(torch.randn((3, 4)))


def test_sampled_forward(device: torch.device) -> None:
    """Test _sampled_forward."""

    class Test(VIModule):
        def __init__(self, ref: Tensor) -> None:
            super().__init__()
            self.ref = ref
            self._log_probs = []

        def forward(self, x: Tensor) -> Tensor:
            assert x.shape == self.ref.shape
            self._log_probs.append(get_unwrapped(torch.randn(2, device=x.device)))
            return x - self.ref

    shape1 = (3, 4)
    sample1 = torch.randn(shape1, device=device)
    test1 = Test(ref=sample1)
    cons, rand = test1.sampled_forward(sample1, samples=10)
    assert torch.allclose(cons, torch.zeros((10,) + shape1, device=device))
    for r in rand[1:]:
        assert not torch.allclose(rand[0], r)

    shape2 = (5,)
    sample2 = torch.randn(shape2, device=device)
    test2 = Test(ref=sample2)
    assert torch.allclose(
        test2.sampled_forward(sample2, samples=1)[0],
        torch.zeros((1,) + shape2, device=device),
    )


def test_name_maker() -> None:
    """Test VIModule.variational_parameter_name."""
    assert VIModule.variational_parameter_name("a", "b") == "_a_b"
    assert VIModule.variational_parameter_name("vw", "xz") == "_vw_xz"


def test_vimodule(device: torch.device) -> None:
    """Test VIModule."""

    # Test variant without parameters
    class DummyModule(VIModule):
        pass

    module1 = DummyModule()

    assert module1.random_variables is None
    assert module1.return_log_probs
    assert module1._has_sampling_responsibility
    assert not hasattr(module1, "variational_distribution")
    assert not hasattr(module1, "prior")
    assert not hasattr(module1, "_kaiming_init")
    assert not hasattr(module1, "_rescale_prior")
    assert not hasattr(module1, "_prior_init")

    with pytest.raises(
        NoVariablesError, match="DummyModule has no random variables to reset"
    ):
        module1.reset_parameters()
    with pytest.raises(
        NoVariablesError, match="DummyModule has no variational parameters to get"
    ):
        module1.get_variational_parameters("weight")
    with pytest.raises(NoVariablesError, match="DummyModule has no random variables"):
        module1.get_log_probs(
            [
                torch.zeros(1),
            ]
        )
    with pytest.raises(
        NoVariablesError, match="DummyModule has no random variables to sample"
    ):
        module1.sample_variables()

    # Test variant with parameters
    var_dict1 = dict(
        weight=(2, 3),
        bias=(3,),
    )
    var_params = ("mean", "std")
    default_params = (0.0, 0.3)

    class TestDistribution(VariationalDistribution):
        variational_parameters = var_params
        _default_variational_parameters = default_params

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            pass

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            pass

    class TestPrior(Prior):
        distribution_parameters = ("mean", "std")
        _scaling_parameters = ("mean", "std")
        mean: float = 1.0
        std: float = 2.0

        def log_prob(self, x: Tensor) -> Tensor:
            pass

    module2 = VIModule(var_dict1, TestDistribution(), TestPrior(), device=device)

    for var in var_dict1:
        for param in var_params:
            param_name = module2.variational_parameter_name(var, param)
            assert hasattr(module2, param_name)
            assert getattr(module2, param_name).device == device
            if param != "mean":
                # kaiming_init scales with sqrt(fan_in=3)
                scale = 1 / math.sqrt(3)
                index = var_params.index(param)
                default = default_params[index]
                assert (getattr(module2, param_name) == default * scale).all()

    # Check that reset_mean randomizes the means
    weight_mean = module2._weight_mean.clone()
    bias_mean = module2._bias_mean.clone()

    module2.reset_parameters()

    assert not (module2._weight_mean == weight_mean).all()
    assert not (module2._bias_mean == bias_mean).all()

    # Test prior based initialization
    with pytest.warns(
        UserWarning,
        match=r'Module \[TestPrior\] is missing the "reset_parameters" method*',
    ):
        _ = VIModule(
            var_dict1,
            TestDistribution(),
            TestPrior(),
            prior_initialization=True,
            device=device,
        )

    with pytest.raises(
        AssertionError,
        match=r"Provide either exactly one variational distribution or exactly one for each random variable",
    ):
        _ = VIModule(var_dict1, [TestDistribution()] * 3, TestPrior(), device=device)

    with pytest.raises(
        AssertionError,
        match=r"Provide either exactly one prior distribution or exactly one for each random variable",
    ):
        _ = VIModule(var_dict1, TestDistribution(), [TestPrior()] * 3, device=device)

    _ = VIModule(var_dict1, [TestDistribution()] * 2, [TestPrior()] * 2, device=device)

    filterwarnings("error")
    module2 = VIModule(
        var_dict1, TestDistribution(), TestPrior(), rescale_prior=True, device=device
    )
    for prior in module2.prior:
        assert prior.mean == 1 / math.sqrt(3 * 3)  # type: ignore [attr-defined]
        assert prior.std == 2 / math.sqrt(3 * 3)  # type: ignore [attr-defined]


def test_get_variational_parameters(device: torch.device) -> None:
    """Test VIBaseModule.get_variational_parameters."""
    var_dict1 = dict(
        weight=(2, 3),
        bias=(3,),
    )
    var_params = ("mean", "log_std")
    default_params = (0.0, 0.3)

    class TestDistribution(VariationalDistribution):
        variational_parameters = var_params
        _default_variational_parameters = default_params

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            pass

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            pass

    class TestPrior(Prior):
        distribution_parameters = ("mean", "log_std")
        _scaling_parameters = ()

        def log_prob(self, x: Tensor) -> Tensor:
            pass

    module = VIModule(var_dict1, TestDistribution(), TestPrior(), device=device)

    for variable in ("weight", "bias"):
        params_list = module.get_variational_parameters(variable)
        for param_name, param_value in zip(var_params, params_list):
            assert param_value.shape == var_dict1[variable]
            assert param_value.device == device
            assert (
                param_value
                == getattr(
                    module, module.variational_parameter_name(variable, param_name)
                )
            ).all()


def test_get_log_probs(device: torch.device) -> None:
    """Test VIBaseModule.get_log_probs."""
    var_dict1 = dict(
        weight=(3, 1),
        bias=(1,),
    )
    var_params = ("mean", "log_std")
    default_params = (0.0, 0.3)

    class TestDistribution(VariationalDistribution):
        variational_parameters = var_params
        _default_variational_parameters = default_params

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            pass

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            return torch.tensor(3.0)

    class TestPrior(Prior):
        distribution_parameters = ("mean", "log_std")
        _scaling_parameters = ()

        def log_prob(self, x: Tensor) -> Tensor:
            return torch.tensor(2.0, device=x.device)

    module = VIModule(var_dict1, TestDistribution(), TestPrior(), device=device)
    module.random_variables = cast(Tuple[str, ...], module.random_variables)
    params = [torch.empty(1, device=device)] * len(module.random_variables)
    prior_log_prob, variational_log_prob = module.get_log_probs(params)

    assert prior_log_prob == 2.0 * len(module.random_variables)
    assert variational_log_prob == 3.0 * len(module.random_variables)


def test_log_prob_setting(device: torch.device) -> None:
    """Test setting of _return_log_probs with VIModule.return_log_probs."""
    from torch_bayesian.vi import VILinear

    in_features = 3
    out_features = 5

    class Test(VIModule):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.module = VILinear(d_in, d_out, device=device)

        def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
            return self.module(x)

    module1 = Test(in_features, out_features)
    module1.return_log_probs = True
    assert module1._return_log_probs is True
    assert module1.module._return_log_probs is True
    sample1 = torch.randn(4, in_features, device=device)
    out = module1(sample1, samples=10)
    assert len(out) == 2
    assert out[0].shape == (10, 4, out_features)
    assert out[0].device == device
    assert out[1].shape == (10, 2)
    assert out[1].device == device

    module1.return_log_probs = False
    assert module1._return_log_probs is False
    assert module1.module._return_log_probs is False
    sample1 = torch.randn(4, in_features, device=device)
    out = module1(sample1, samples=10)
    assert out.shape == (10, 4, out_features)
    assert out.device == device


@pytest.mark.filterwarnings("ignore")
def test_slow_forward(device: torch.device) -> None:
    """
    Test VIModule._slow_forward.

    I'm not familiar with the intricacies of what this method is supposed to do.
    I basically just copied it from torch to maintain all features.
    """
    # Let's just test it by jitifying something

    class Test(VIModule):
        _log_probs: List[Tensor] = []

        def forward(self, x: Tensor) -> Tensor:
            self._log_probs.append(torch.tensor((5.0, 3.0), device=x.device))
            return x

    module1 = Test()
    inputs = torch.randn(4, 3, device=device)
    filterwarnings(
        "ignore",
        "Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!",
    )
    _ = torch.jit.trace(module1.forward, inputs)


def test_hooks(device: torch.device) -> None:
    """Test hook execution by VIModule._call_impl."""

    def backward_pre_hook(module: Module, grad: Tensor) -> None:
        pass

    def backward_hook(module: Module, grad_in: Tensor, grad_out: Tensor) -> None:
        pass

    def forward_pre_hook(module: Module, args: Any) -> Any:
        return torch.tensor(3.0, device=device)

    def forward_hook(module: Module, args: Any, out: Any) -> Any:
        return torch.tensor(3.0, device=device)

    def forward_pre_hook_with_kwargs(
        module: Module, args: Any, kwargs: Any
    ) -> Tuple[Any, Any]:
        return args, kwargs

    def forward_hook_with_kwargs(
        module: Module, args: Any, kwargs: Any, out: Any
    ) -> None:
        pass

    class Test(VIModule):
        _log_probs: List[Tensor] = []

        def forward(self, x: Tensor) -> Tensor:
            self._log_probs.append(get_unwrapped(torch.randn(2, device=x.device)))
            return x

    test = Test()
    test.register_forward_pre_hook(forward_pre_hook)
    test.register_forward_pre_hook(forward_pre_hook_with_kwargs, with_kwargs=True)
    test.register_forward_hook(forward_hook, always_call=True)
    test.register_forward_hook(forward_hook_with_kwargs, with_kwargs=True)
    test.register_full_backward_pre_hook(backward_pre_hook)
    test.register_full_backward_hook(backward_hook)

    inputs = torch.randn(4, 3, device=device)
    _ = test(inputs)
