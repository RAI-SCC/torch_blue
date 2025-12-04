import math
from itertools import product
from typing import Dict, Optional, Tuple, Type, cast

import pytest
import torch
from torch import Tensor
from torch._C._functorch import get_unwrapped

from torch_blue.vi import VIModule, VIReturn
from torch_blue.vi.distributions import Distribution
from torch_blue.vi.utils import NoVariablesError, UnsupportedDistributionError


class TestVIModule:
    """Test core functionality of VIModule."""

    @pytest.fixture(
        params=[(("mean", "std"), (0.0, 1.0)), (("mean", "std"), (1.0, 0.3))]
    )
    def dummy_prior(self, request: pytest.FixtureRequest) -> Type[Distribution]:
        """Create dummy prior for testing."""
        dist_params, default_params = request.param

        class TestPrior(Distribution):
            is_prior: bool = True
            distribution_parameters: Tuple[str, ...] = dist_params
            _scaling_parameters: Tuple[str, ...] = dist_params
            mean: float = default_params[0]
            std: float = default_params[1]

            def prior_log_prob(self, x: Tensor) -> Tensor:
                return 2 * x

        return TestPrior

    @pytest.fixture(
        params=[
            (("mean", "std"), (0.0, 1.0)),
            (("mean", "std"), (1.0, 0.3)),
            (("mean", "log_std"), (0.0, 0.0)),
        ]
    )
    def dummy_vardist(self, request: pytest.FixtureRequest) -> Type[Distribution]:
        """Create dummy vardist for testing."""
        var_params, default_params = request.param

        class TestVarDist(Distribution):
            is_variational_distribution: bool = True
            distribution_parameters: Tuple[str, ...] = var_params
            _default_variational_parameters: Tuple[float, ...] = default_params

            def sample(self, mean: Tensor, std: Tensor) -> Tensor:
                return mean + std

            def variational_log_prob(
                self, sample: Tensor, mean: Tensor, std: Tensor
            ) -> Tensor:
                return 3 * sample

        return TestVarDist

    @pytest.fixture(
        params=[
            {"weight": (5, 6), "bias": (6,)},
            {"weight": (5, 6), "weight2": (3, 2), "bias": (6,)},
            {"weight": (3, 2), "bias": None},
            {"weight": (3, 2)},
        ]
    )
    def var_dict(
        self, request: pytest.FixtureRequest
    ) -> Dict[str, Optional[Tuple[int, ...]]]:
        """Set up variable dict for test initializations."""
        return request.param

    def test_dist_no_error(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test error for incompatible number of distributions."""
        n_var = len(var_dict)
        n_test = [n_var + 1, n_var - 1]

        for n in n_test:
            with pytest.raises(
                AssertionError,
                match=r"Provide either exactly one variational distribution or "
                r"exactly one for each random variable",
            ):
                _ = VIModule(
                    var_dict, [dummy_vardist()] * n, dummy_prior(), device=device
                )

            with pytest.raises(
                AssertionError,
                match=r"Provide either exactly one prior distribution or "
                r"exactly one for each random variable",
            ):
                _ = VIModule(
                    var_dict, dummy_vardist(), [dummy_prior()] * n, device=device
                )

        # Test correct settings
        _ = VIModule(var_dict, dummy_vardist(), [dummy_prior()] * n_var, device=device)
        _ = VIModule(var_dict, [dummy_vardist()] * n_var, dummy_prior(), device=device)
        _ = VIModule(
            var_dict, [dummy_vardist()] * n_var, [dummy_prior()] * n_var, device=device
        )

    def test_prior_init_error(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test error for missing prior init, if requested."""
        with pytest.warns(
            UserWarning,
            match=r'Module \[TestPrior\] is missing the "reset_parameters_to_prior"'
            r" method*",
        ):
            _ = VIModule(
                var_dict,
                dummy_vardist(),
                dummy_prior(),
                prior_initialization=True,
                device=device,
            )

    def test_kaiming_scaling(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test kaiming scaling of non-mean parameters."""
        var_params = dummy_vardist.distribution_parameters
        default_params = dummy_vardist._default_variational_parameters
        module = VIModule(var_dict, dummy_vardist(), dummy_prior(), device=device)
        fan_in = module._calculate_fan_in(var_dict)

        for var in var_dict:
            if var_dict[var] is None:
                # Check None shape variables do not create parameters
                for param in var_params:
                    param_name = module.variational_parameter_name(var, param)
                    assert not hasattr(module, param_name)
                continue

            for param in var_params:
                param_name = module.variational_parameter_name(var, param)
                assert hasattr(module, param_name)
                assert getattr(module, param_name).device == device
                if param != "mean":
                    # kaiming_init scales with sqrt(fan_in)
                    scale = 1 / math.sqrt(fan_in)
                    index = var_params.index(param)
                    default = default_params[index]
                    if param.startswith("log"):
                        assert (
                            getattr(module, param_name)
                            == default + math.log(scale + 1e-5)
                        ).all()
                    else:
                        assert (getattr(module, param_name) == default * scale).all()

    def test_invalid_distribution_error(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test errors if distribution is used in unsupported role."""

        class InvalidVarDist(dummy_prior):  # type: ignore[valid-type,misc]
            pass

        class InvalidPrior(dummy_vardist):  # type: ignore[valid-type,misc]
            pass

        invalid_vardist = InvalidVarDist()
        invalid_prior = InvalidPrior()
        n_var = len(var_dict)

        with pytest.raises(
            UnsupportedDistributionError,
            match="InvalidVarDist does not support use as variational distribution.",
        ):
            _ = VIModule(
                var_dict, [invalid_vardist] * n_var, dummy_prior(), device=device
            )

        with pytest.raises(
            UnsupportedDistributionError,
            match="InvalidVarDist does not support use as variational distribution.",
        ):
            _ = VIModule(var_dict, invalid_vardist, dummy_prior(), device=device)

        with pytest.raises(
            UnsupportedDistributionError,
            match="InvalidPrior does not support use as prior.",
        ):
            _ = VIModule(var_dict, dummy_vardist(), invalid_prior, device=device)

        with pytest.raises(
            UnsupportedDistributionError,
            match="InvalidPrior does not support use as prior.",
        ):
            _ = VIModule(
                var_dict, dummy_vardist(), [invalid_prior] * n_var, device=device
            )

    def test_no_variables_error(
        self,
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test error if all variables are set as None."""
        with pytest.raises(
            NoVariablesError, match="All module variables are set to None."
        ):
            _ = VIModule(
                dict(a=None),
                dummy_vardist(),
                dummy_prior(),
                device=device,
            )

    def test_prior_rescaling(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test rescaling prior."""
        ref_mean = dummy_prior.mean  # type: ignore[attr-defined]
        ref_std = dummy_prior.std  # type: ignore[attr-defined]

        module = VIModule(
            var_dict, dummy_vardist(), dummy_prior(), rescale_prior=True, device=device
        )
        fan_in = module._calculate_fan_in(var_dict)

        for prior in module.prior.values():
            if prior is None:
                continue
            assert prior.mean == ref_mean / math.sqrt(3 * fan_in)  # type: ignore [attr-defined]
            assert prior.std == ref_std / math.sqrt(3 * fan_in)  # type: ignore [attr-defined]

    def test_parameter_resetting(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test reset parameters."""
        # Means should be randomized, everything else not
        module = VIModule(var_dict, dummy_vardist(), dummy_prior(), device=device)

        memory = dict()
        for var in var_dict:
            if var_dict[var] is None:
                continue
            for param in dummy_vardist.distribution_parameters:
                name = module.variational_parameter_name(var, param)
                memory[name] = (getattr(module, name).clone(), param != "mean")

        module.reset_variational_parameters()

        for name in memory:
            assert (memory[name][0] == getattr(module, name)).all() == memory[name][1]

    def test_get_variational_parameters(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test VIModule.get_variational_parameters."""
        var_params = dummy_vardist.distribution_parameters
        variables = list(var_dict.keys())

        module = VIModule(var_dict, dummy_vardist(), dummy_prior(), device=device)

        for variable in variables:
            if var_dict[variable] is None:
                with pytest.raises(
                    NoVariablesError,
                    match=f"{variable} is None and has no parameters to get",
                ):
                    _ = module.get_variational_parameters(variable)
                continue
            params_list = module.get_variational_parameters(variable)
            for param_name, param_value in zip(var_params, params_list):
                name = module.variational_parameter_name(variable, param_name)
                assert param_value.shape == var_dict[variable]
                assert param_value.device == device
                assert (param_value == getattr(module, name)).all()

    def test_get_log_probs(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        device: torch.device,
    ) -> None:
        """Test VIModule.get_log_probs."""
        module = VIModule(var_dict, dummy_vardist(), dummy_prior(), device=device)
        assert module.random_variables is not None

        for variable in module.random_variables:
            if var_dict[variable] is None:
                continue
            params = torch.empty(1, device=device)
            prior_log_prob, variational_log_prob = module.get_log_probs(
                params, variable
            )

            assert prior_log_prob == 2 * params
            assert variational_log_prob == 3 * params

    @pytest.mark.parametrize("return_log_probs", [True, False])
    def test_sample_variable(
        self,
        var_dict: Dict[str, Optional[Tuple[int, ...]]],
        dummy_prior: Type[Distribution],
        dummy_vardist: Type[Distribution],
        return_log_probs: bool,
        device: torch.device,
    ) -> None:
        """Test VIModule.get_sample_variable."""
        module = VIModule(var_dict, dummy_vardist(), dummy_prior(), device=device)
        module.return_log_probs = return_log_probs

        for variable in var_dict:
            sample = getattr(module, variable)
            if var_dict[variable] is None:
                assert sample is None
                continue
            params = module.get_variational_parameters(variable)
            assert torch.allclose(sample, torch.add(*params))

            if return_log_probs:
                lps = module.gather_log_probs()
                assert torch.allclose(lps[1] / lps[0], torch.tensor(3 / 2))


class DummyModule1(VIModule):
    """Dummy module for testing."""

    def __init__(self, ref: Tensor, device: torch.device) -> None:
        super().__init__()
        self.ref = torch.tensor(False, device=device) if ref is None else ref
        self._log_probs = dict(all=[])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        assert x.shape == self.ref.shape
        self._log_probs["all"].append(get_unwrapped(torch.randn(2, device=x.device)))
        return x.to(torch.float), -self.ref.to(torch.float)


class DummyModule2(VIModule):
    """Additional dummy module with wrapper and unused module for testing."""

    def __init__(self, ref: Tensor, device: torch.device) -> None:
        super().__init__()
        self.module = DummyModule1(ref, device)
        self.vestigial = DummyModule1(ref, device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        a, b = self.module(x)
        return a + b


@pytest.mark.parametrize("shape", [(3, 4), (5,), None])
class TestVIModuleShapeDependents:
    """Test sample shape dependent features of VIModule."""

    def test_expand_to_samples(
        self, shape: Optional[Tuple[int, ...]], device: torch.device
    ) -> None:
        """Test _expand_to_samples."""
        n_samples = torch.randint(1, 10, [1]).item()

        if shape is not None:
            sample = torch.randn(shape, device=device)
            out = VIModule._expand_to_samples(sample, samples=n_samples)
            assert out.shape == (n_samples,) + shape
            for s in out:
                assert (s == sample).all()
        else:
            out = VIModule._expand_to_samples(None, samples=n_samples)
            assert out.shape == (n_samples,)
            for s in out:
                assert s.item() is False

    def test_no_forward_error(
        self, shape: Optional[Tuple[int, ...]], device: torch.device
    ) -> None:
        """Test that forward throws error if not implemented."""
        module = VIModule()
        sample = None if shape is None else torch.randn(shape, device=device)
        with pytest.raises(
            NotImplementedError,
            match=r'Module \[VIModule\] is missing the required "forward" function',
        ):
            module.forward(sample)

    @pytest.mark.parametrize(
        "module,log_probs", product([DummyModule1, DummyModule2], [True, False])
    )
    def test_sampled_forward(
        self,
        shape: Optional[Tuple[int, ...]],
        module: Type[DummyModule1],
        log_probs: bool,
        device: torch.device,
    ) -> None:
        """Test _sampled_forward."""
        n_samples = torch.randint(1, 10, [1]).item()

        sample = None if shape is None else torch.randn(shape, device=device)
        test = module(ref=sample, device=device)
        test.return_log_probs = log_probs
        out = cast(VIReturn, test(sample, samples=n_samples))

        if isinstance(out, tuple):
            lps = cast(Tensor, out[0].log_probs)
            for r in out:
                assert hasattr(r, "log_probs")
                assert isinstance(r, VIReturn)
                assert r.log_probs is lps
            out = out[0] + out[1]
        else:
            lps = cast(Tensor, out.log_probs)
            assert hasattr(out, "log_probs")
            assert isinstance(out, VIReturn)

        if log_probs:
            assert lps.shape == (n_samples, 2)
            if shape is not None:
                assert torch.allclose(
                    out, torch.zeros((n_samples,) + shape, device=device)
                )
            for r in lps[1:]:
                assert not torch.allclose(lps[0], r)
        else:
            assert lps is None


class TestVIModuleGeneral:
    """Test general funtionalities of VIModule."""

    @pytest.mark.parametrize("var,param", product(["a", "vw"], ["b", "xy"]))
    def test_name_maker(self, var: str, param: str) -> None:
        """Test VIModule.variational_parameter_name."""
        target = "_".join(["", var, param])
        assert VIModule.variational_parameter_name(var, param) == target

    def test_trivial_init(self) -> None:
        """Test init for parameter-free modules."""

        class DummyModule(VIModule):
            pass

        module1 = DummyModule()

        assert module1.random_variables is None
        assert module1.return_log_probs
        assert module1._has_sampling_responsibility
        assert not hasattr(module1, "variational_distribution")
        assert not hasattr(module1, "prior")
        assert not hasattr(module1, "_kaiming_init")
        assert not hasattr(module1, "_prior_init")

        with pytest.raises(
            NoVariablesError, match="DummyModule has no random variables to reset"
        ):
            module1.reset_variational_parameters()
        with pytest.raises(
            NoVariablesError, match="DummyModule has no variational parameters to get"
        ):
            module1.get_variational_parameters("weight")
        with pytest.raises(
            NoVariablesError, match="DummyModule has no random variables"
        ):
            module1.get_log_probs(torch.zeros(1), "foo")
        with pytest.raises(
            NoVariablesError, match="DummyModule has no random variables to sample"
        ):
            module1.sample_variable("foo")
