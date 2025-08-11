import random
from math import log, pi
from typing import Optional, Type, Union

import pytest
import torch
from torch import Tensor

from torch_bayesian.vi import (
    AnalyticalKullbackLeiblerLoss,
    KullbackLeiblerLoss,
    KullbackLeiblerModule,
    VILinear,
    VIModule,
    VISequential,
)
from torch_bayesian.vi.analytical_kl_loss import (
    NonBayesianDivergence,
    NormalNormalDivergence,
    UniformNormalDivergence,
)
from torch_bayesian.vi.predictive_distributions import (
    MeanFieldNormalPredictiveDistribution,
    NonBayesianPredictiveDistribution,
    PredictiveDistribution,
)
from torch_bayesian.vi.priors import MeanFieldNormalPrior, Prior, UniformPrior
from torch_bayesian.vi.utils import use_norm_constants
from torch_bayesian.vi.variational_distributions import (
    MeanFieldNormalVarDist,
    NonBayesian,
    VariationalDistribution,
)


def test_klmodule(device: torch.device) -> None:
    """Test KullbackLeiblerModule."""
    module = KullbackLeiblerModule()

    list_a = [torch.arange(27, 31, device=device)]
    list_b = [torch.arange(3, device=device)]
    reference = torch.cat([*list_a, *list_b])

    with pytest.raises(NotImplementedError):
        module(list_a, list_b)

    def dummy_forward(*args: Tensor) -> Tensor:
        return torch.cat(args)

    module.forward = dummy_forward

    out = module(list_a, list_b)
    assert out.device == device
    assert torch.allclose(out, reference)


def test_nonbayesian_klmodule(device: torch.device) -> None:
    """Test NonBayesianDivergence."""
    sample_shape = [7, 10]
    prior_param_number = random.randint(1, 5)
    var_param_number = 1

    prior_params = [*torch.randn([prior_param_number, *sample_shape], device=device)]
    var_params = [*torch.randn([var_param_number, *sample_shape], device=device)]

    module = NonBayesianDivergence()
    out = module(prior_params, var_params)
    assert out.device == device
    assert torch.allclose(out, torch.tensor([0.0], device=device))


@pytest.mark.parametrize("norm_constants", [(True,), (False,)])
def test_uniformnormal_klmodule(norm_constants: bool, device: torch.device) -> None:
    """Test UniformNormalDivergence."""
    sample_shape = [7, 10]
    prior_param_number = 0
    var_param_number = 2

    prior_params = [*torch.randn([prior_param_number, *sample_shape], device=device)]
    var_params = [*torch.randn([var_param_number, *sample_shape], device=device)]

    use_norm_constants(norm_constants)
    module = UniformNormalDivergence()
    out = module(prior_params, var_params)
    if not norm_constants:
        reference = -var_params[1] - 0.5
    else:
        reference = -var_params[1] - 0.5 * (1 + log(2 * pi))

    assert out.device == device
    assert torch.allclose(out, reference.sum())


@pytest.mark.parametrize("norm_constants", [(True,), (False,)])
def test_normalnormal_klmodule(norm_constants: bool, device: torch.device) -> None:
    """Test NormalNormalDivergence."""
    sample_shape = [7, 10]
    prior_param_number = 2
    var_param_number = 2

    prior_params = [*torch.randn([prior_param_number, *sample_shape], device=device)]
    var_params = [*torch.randn([var_param_number, *sample_shape], device=device)]

    use_norm_constants(norm_constants)
    module = NormalNormalDivergence()
    out = module(prior_params, var_params)

    prior_variance = torch.exp(2 * prior_params[1])
    variational_variance = torch.exp(2 * var_params[1])

    reference = (
        (prior_params[1] - var_params[1])
        + (prior_params[0] - var_params[0]).pow(2) / (2 * prior_variance)
        + variational_variance / (2 * prior_variance)
        - 1 / 2
    )
    assert out.device == device
    assert torch.allclose(out, reference.sum())


class DummyPrior(Prior):
    """Dummy prior for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.distribution_parameters = ()

    def log_prob(self, *args: Tensor) -> Tensor:
        """Return dummy log probability."""
        return torch.zeros(1, device=args[0].device)


class DummyVarDist(VariationalDistribution):
    """Dummy variational distribution for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.variational_parameters = ("mean",)
        self._default_variational_parameters = (0.0,)

    def sample(self, mean: Tensor) -> Tensor:
        """Return dummy sample."""
        return torch.zeros(1, device=mean.device)

    def log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
        """Return dummy log probability."""
        return torch.zeros(1, device=mean.device)


@pytest.mark.parametrize(
    "prior,var_dist,target",
    [
        (MeanFieldNormalPrior, MeanFieldNormalVarDist, NormalNormalDivergence),
        (UniformPrior, MeanFieldNormalVarDist, UniformNormalDivergence),
        (MeanFieldNormalPrior, NonBayesian, NonBayesianDivergence),
        (UniformPrior, NonBayesian, NonBayesianDivergence),
        (DummyPrior, MeanFieldNormalVarDist, "fail"),
        (MeanFieldNormalPrior, DummyVarDist, "fail"),
    ],
)
def test_detect_divergence(
    prior: Type[Prior],
    var_dist: Type[VariationalDistribution],
    target: Union[str, Type[KullbackLeiblerModule]],
) -> None:
    """Test AnalyticalKullbackLeiblerLoss._detect_divergence()."""
    if target == "fail":
        with pytest.raises(
            NotImplementedError,
            match=f"Analytical loss is not implemented for {prior.__name__} and {var_dist.__name__}.",
        ):
            AnalyticalKullbackLeiblerLoss._detect_divergence(prior(), var_dist())
    elif not isinstance(target, str):
        assert isinstance(
            AnalyticalKullbackLeiblerLoss._detect_divergence(prior(), var_dist()),
            target,
        )
    else:
        raise ValueError("Invalid target specification.")


class DummyMLP(VIModule):
    """Dummy MLP for testing."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        prior: Prior = MeanFieldNormalPrior(),
        var_dist: VariationalDistribution = MeanFieldNormalVarDist(),
        alt_prior: Optional[Prior] = None,
        alt_vardist: Optional[VariationalDistribution] = None,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        alt_prior = alt_prior or prior
        alt_var_dist = alt_vardist or var_dist

        self.layers = VISequential(
            VILinear(
                in_features,
                hidden_features,
                variational_distribution=var_dist,
                prior=prior,
                return_log_probs=return_log_probs,
                device=device,
            ),
            VILinear(
                hidden_features,
                out_features,
                variational_distribution=alt_var_dist,
                prior=alt_prior,
                return_log_probs=return_log_probs,
                device=device,
            ),
        )

    def forward(self, input_: Tensor) -> Tensor:
        """Make forward pass."""
        return self.layers(input_)


@pytest.mark.parametrize(
    "prior,var_dist,norm_constants",
    [
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), True),
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), False),
        (MeanFieldNormalPrior(1.0, 1.5), MeanFieldNormalVarDist(0.5), True),
        (MeanFieldNormalPrior(1.0, 1.5), MeanFieldNormalVarDist(0.5), False),
        (UniformPrior(), NonBayesian(), False),
        (UniformPrior(), NonBayesian(), True),
        (UniformPrior(), MeanFieldNormalVarDist(), False),
        (UniformPrior(), MeanFieldNormalVarDist(), True),
        (UniformPrior(), MeanFieldNormalVarDist(0.7), False),
        (UniformPrior(), MeanFieldNormalVarDist(0.7), True),
    ],
)
def test_prior_matching(
    prior: Prior,
    var_dist: VariationalDistribution,
    norm_constants: bool,
    device: torch.device,
) -> None:
    """Test AnalyticalKullbackLeiblerLoss.prior_matching()."""
    use_norm_constants(norm_constants)

    f_in = 8
    f_hidden = 16
    f_out = 10

    batch_size = 100
    samples = 10000

    model = DummyMLP(
        f_in, f_hidden, f_out, prior=prior, var_dist=var_dist, device=device
    )
    criterion = AnalyticalKullbackLeiblerLoss(
        model, MeanFieldNormalPredictiveDistribution(), samples
    )
    ref_criterion = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), samples, track=True
    )

    sample = torch.rand([batch_size, f_in], device=device)
    target = torch.rand([batch_size, f_out], device=device)

    model.return_log_probs = True

    out = model(sample, samples=samples)

    analytical_prior_matching = criterion.prior_matching()
    ref_criterion(out, target)
    ref_prior_matching = ref_criterion.log["prior_matching"]  # type: ignore [index]
    assert analytical_prior_matching.device == device
    assert torch.allclose(
        torch.tensor(ref_prior_matching[0]),
        analytical_prior_matching,
        atol=5e-1,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "prior,var_dist,alt_prior,alt_var_dist,target_kl_module,heat,dataset_size,divergence_type,track,expected_error",
    [
        # Test base parameters
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            None,
            None,
            NormalNormalDivergence,
            None,
            None,
            None,
            False,
            None,
        ),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            None,
            None,
            NormalNormalDivergence,
            None,
            None,
            None,
            True,
            None,
        ),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            None,
            None,
            NormalNormalDivergence,
            0.2,
            None,
            None,
            False,
            None,
        ),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            None,
            None,
            NormalNormalDivergence,
            None,
            512,
            None,
            False,
            None,
        ),
        # Test autodetection for each Prior-Vardist combination
        (
            UniformPrior(),
            MeanFieldNormalVarDist(),
            None,
            None,
            UniformNormalDivergence,
            None,
            None,
            None,
            False,
            None,
        ),
        (
            MeanFieldNormalPrior(),
            NonBayesian(),
            None,
            None,
            NonBayesianDivergence,
            None,
            None,
            None,
            False,
            None,
        ),
        (
            UniformPrior(),
            NonBayesian(),
            None,
            None,
            NonBayesianDivergence,
            None,
            None,
            None,
            False,
            None,
        ),
        # Test error raising
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            UniformPrior(),
            None,
            NormalNormalDivergence,
            None,
            None,
            None,
            False,
            0,
        ),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            None,
            NonBayesian(),
            NormalNormalDivergence,
            None,
            None,
            None,
            False,
            0,
        ),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            UniformPrior(),
            NonBayesian(),
            NormalNormalDivergence,
            None,
            None,
            None,
            False,
            0,
        ),
        (
            DummyPrior(),
            MeanFieldNormalVarDist(),
            None,
            None,
            NormalNormalDivergence,
            None,
            None,
            None,
            False,
            1,
        ),
        (
            MeanFieldNormalPrior(),
            DummyVarDist(),
            None,
            None,
            NormalNormalDivergence,
            None,
            None,
            None,
            False,
            1,
        ),
        (
            DummyPrior(),
            DummyVarDist(),
            None,
            None,
            NormalNormalDivergence,
            None,
            None,
            None,
            False,
            1,
        ),
        (None, None, None, None, NormalNormalDivergence, None, None, None, False, 2),
        # Test error overwriting by manual KL-Module specification
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            UniformPrior(),
            None,
            NormalNormalDivergence,
            None,
            None,
            NormalNormalDivergence(),
            False,
            None,
        ),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            None,
            NonBayesian(),
            UniformNormalDivergence,
            None,
            None,
            UniformNormalDivergence(),
            False,
            None,
        ),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            UniformPrior(),
            NonBayesian(),
            NonBayesianDivergence,
            None,
            None,
            NonBayesianDivergence(),
            False,
            None,
        ),
        (
            DummyPrior(),
            MeanFieldNormalVarDist(),
            None,
            None,
            NormalNormalDivergence,
            None,
            None,
            NormalNormalDivergence(),
            False,
            None,
        ),
        (
            MeanFieldNormalPrior(),
            DummyVarDist(),
            None,
            None,
            UniformNormalDivergence,
            None,
            None,
            UniformNormalDivergence(),
            False,
            None,
        ),
        (
            DummyPrior(),
            DummyVarDist(),
            None,
            None,
            NonBayesianDivergence,
            None,
            None,
            NonBayesianDivergence(),
            False,
            None,
        ),
    ],
)
def test_init(
    prior: Prior,
    var_dist: VariationalDistribution,
    alt_prior: Optional[Prior],
    alt_var_dist: Optional[VariationalDistribution],
    target_kl_module: Type[KullbackLeiblerModule],
    heat: Optional[float],
    dataset_size: Optional[int],
    divergence_type: Optional[KullbackLeiblerModule],
    track: bool,
    expected_error: Optional[int],
    device: torch.device,
) -> None:
    """Test AnalyticalKullbackLeiblerLoss initialization."""
    f_in = 8
    f_hidden = 16
    f_out = 10

    predictive_distribution = MeanFieldNormalPredictiveDistribution()
    alt_prior = alt_prior or prior
    alt_var_dist = alt_var_dist or var_dist
    target_heat = heat or 1.0

    if prior is None and var_dist is None:
        model = torch.nn.Linear(f_in, f_out, device=device)
    else:
        model = DummyMLP(
            f_in,
            f_hidden,
            f_out,
            prior=prior,
            var_dist=var_dist,
            alt_prior=alt_prior,
            alt_vardist=alt_var_dist,
            return_log_probs=True,
            device=device,
        )

    kwargs = dict(
        model=model, predictive_distribution=predictive_distribution, track=track
    )
    if dataset_size is not None:
        kwargs["dataset_size"] = dataset_size
    if heat is not None:
        kwargs["heat"] = heat
    if divergence_type is not None:
        kwargs["divergence_type"] = divergence_type

    if expected_error is not None:
        error_list = [
            (
                NotImplementedError,
                "Handling of inconsistent distributions types is not implemented yet.",
            ),
            (
                NotImplementedError,
                f"Analytical loss is not implemented for {prior.__class__.__name__} and {var_dist.__class__.__name__}.",
            ),
            (ValueError, "Provided model is not bayesian."),
        ]
        error, message = error_list[expected_error]
        with pytest.raises(error, match=message):
            AnalyticalKullbackLeiblerLoss(**kwargs)  # type: ignore [arg-type]
        return

    criterion = AnalyticalKullbackLeiblerLoss(**kwargs)  # type: ignore [arg-type]

    assert criterion.predictive_distribution == predictive_distribution
    assert criterion.dataset_size == dataset_size
    assert criterion.heat == target_heat
    assert criterion.model == model
    assert not model._return_log_probs

    assert criterion._track == track
    if track:
        assert criterion.log is not None
        for key in ["data_fitting", "prior_matching"]:
            assert criterion.log[key] == []
            assert criterion.log[key] == []
        criterion.track(False)
        assert not criterion._track
    else:
        assert criterion.log is None
        criterion.track()
        assert criterion._track
        assert criterion.log is not None
        for key in ["data_fitting", "prior_matching"]:
            assert criterion.log[key] == []
            assert criterion.log[key] == []


@pytest.mark.parametrize(
    "prior,var_dist,heat,init_dataset_size,fwrd_dataset_size,track,norm_constants",
    [
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), 1.0, 50, None, False, False),
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), 1.0, 50, None, True, False),
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), 0.5, 50, None, False, False),
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), 1.0, 50, 25, False, False),
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), 1.0, None, 25, False, False),
        (
            MeanFieldNormalPrior(),
            MeanFieldNormalVarDist(),
            1.0,
            None,
            None,
            False,
            False,
        ),
        (MeanFieldNormalPrior(), MeanFieldNormalVarDist(), 1.0, 50, None, False, True),
        (UniformPrior(), MeanFieldNormalVarDist(), 1.0, 50, None, False, False),
        (UniformPrior(), MeanFieldNormalVarDist(), 1.0, 50, None, False, True),
        (UniformPrior(), NonBayesian(), 1.0, 50, None, False, False),
        (UniformPrior(), NonBayesian(), 1.0, 50, None, False, True),
    ],
)
def test_forward(
    prior: Prior,
    var_dist: VariationalDistribution,
    heat: float,
    init_dataset_size: Optional[int],
    fwrd_dataset_size: Optional[int],
    track: bool,
    norm_constants: bool,
    device: torch.device,
) -> None:
    """Test AnalyticalKullbackLeiblerLoss.forward()."""
    use_norm_constants(norm_constants)

    f_in = 8
    f_hidden = 16
    f_out = 10

    batch_size = 100
    samples = 5000
    test_epochs = 3

    model = DummyMLP(
        f_in, f_hidden, f_out, prior=prior, var_dist=var_dist, device=device
    )
    if isinstance(var_dist, NonBayesian):
        predictive_distribution: Type[PredictiveDistribution] = (
            NonBayesianPredictiveDistribution
        )
    else:
        predictive_distribution = MeanFieldNormalPredictiveDistribution

    criterion = AnalyticalKullbackLeiblerLoss(
        model,
        predictive_distribution(),
        init_dataset_size,
        heat=heat,
        track=track,
    )
    ref_criterion = KullbackLeiblerLoss(
        predictive_distribution(),
        init_dataset_size,
        heat=heat,
        track=True,
    )

    sample = torch.rand([batch_size, f_in], device=device)
    target = torch.rand([batch_size, f_out], device=device)
    optimizer = torch.optim.Adam(model.parameters())

    model.return_log_probs = True

    for _ in range(test_epochs):
        output = model(sample, samples=samples)

        if init_dataset_size is None and fwrd_dataset_size is None:
            message = f"No dataset_size is provided. Batch size \\({batch_size}\\) is used instead."
            with pytest.warns(UserWarning, match=message):
                analytical_loss = criterion(output[0], target, fwrd_dataset_size)
            ref_loss = ref_criterion(output, target, batch_size)
        else:
            analytical_loss = criterion(output[0], target, fwrd_dataset_size)
            ref_loss = ref_criterion(output, target, fwrd_dataset_size)

        assert analytical_loss.device == device
        print(analytical_loss, ref_loss)
        assert torch.allclose(analytical_loss, ref_loss, atol=2e-1, rtol=1e-3)

        model.zero_grad()
        analytical_loss.backward()
        optimizer.step()

    if track:
        for item, ref in zip(
            criterion.log["data_fitting"],  # type: ignore [index]
            ref_criterion.log["data_fitting"],  # type: ignore [index]
        ):
            assert item == ref
        for item, ref in zip(
            criterion.log["prior_matching"],  # type: ignore [index]
            ref_criterion.log["prior_matching"],  # type: ignore [index]
        ):
            assert torch.allclose(
                torch.tensor(item), torch.tensor(ref), atol=2e-1, rtol=1e-3
            )
