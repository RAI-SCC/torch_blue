from itertools import product
from math import log, prod
from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor, nn
from torch import distributions as dist

from torch_blue.vi.distributions import MeanFieldNormal, Normal
from torch_blue.vi.distributions.normal import CorrelatedNormal
from torch_blue.vi.utils import use_norm_constants


class TestNormal:
    """Tests for Normal distribution."""

    target = Normal

    @pytest.mark.parametrize(
        "norm_constants,params", list(product([True, False], [None, (0.7, 0.3, 1e-3)]))
    )
    def test_prior_log_prob(
        self,
        norm_constants: bool,
        params: Optional[Tuple[float, float, float]],
        device: torch.device,
    ) -> None:
        """Test Normal.prior_log_prob."""
        use_norm_constants(norm_constants)

        if params is None:
            prior = self.target()
            assert prior.std == 1.0
            mean = 0.0
            std = 1.0
            eps = 0.0
        else:
            prior = self.target(*params)
            mean, std, eps = params

        ref_dist = dist.Normal(mean, std)

        shape = (3, 4)
        sample = ref_dist.sample(shape).to(device=device)
        ref = -0.5 * (2 * log(std) + (sample - mean) ** 2 / (std**2 + eps))
        if norm_constants:
            norm_const = torch.full(shape, 2 * torch.pi, device=device).log() / 2
            ref -= norm_const
        prior_log_prob = prior.prior_log_prob(sample)
        assert torch.allclose(ref, prior_log_prob, atol=1e-7)

    @pytest.mark.parametrize("norm_constants", [True, False])
    def test_variational_log_prob(
        self, norm_constants: bool, device: torch.device
    ) -> None:
        """Test Normal.variational_log_prob."""
        shape = (3, 4)
        var_dist = self.target()
        use_norm_constants(norm_constants)

        mean = torch.randn(shape, device=device)
        log_std = torch.randn(shape, device=device)

        ref_dist = dist.Normal(mean, log_std.exp())

        sample = ref_dist.sample(shape).to(device=device)
        ref = dist.Normal(mean, log_std.exp()).log_prob(sample).to(device=device)
        if not norm_constants:
            norm_const = torch.full_like(mean, 2 * torch.pi, device=device).log() / 2
            ref += norm_const
        variational_log_prob = var_dist.variational_log_prob(sample, mean, log_std)
        assert torch.allclose(ref, variational_log_prob, atol=1e-7)
        assert variational_log_prob.device == device

    def test_normal_prior_reset(self, device: torch.device) -> None:
        """Test Normal.reset_parameters_to_prior()."""
        param_shape = (5, 4)
        mean = 3.0
        std = 2.0

        class ModuleDummy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight_mean = nn.Parameter(torch.empty(param_shape))
                self.weight_log_std = nn.Parameter(torch.empty(param_shape))

            @staticmethod
            def variational_parameter_name(variable: str, parameter: str) -> str:
                return f"{variable}_{parameter}"

        prior = self.target(mean, std)
        dummy = ModuleDummy().to(device=device)

        iter1 = dummy.parameters()
        prior.reset_parameters_to_prior(dummy, "weight")

        param_mean = iter1.__next__().clone()
        param_log_std = iter1.__next__().clone()

        assert (param_mean == mean).all()
        assert (param_log_std == log(std)).all()
        assert param_mean.device == device
        assert param_log_std.device == device

    def test_normal_sample(self, device: torch.device) -> None:
        """Test _normal_sample."""
        mean = torch.randn((3, 4), device=device)
        std = torch.zeros_like(mean, device=device)
        sample = self.target._normal_sample(mean, std)
        assert sample.shape == mean.shape
        assert torch.allclose(sample, mean)
        assert sample.device == device

        std = torch.ones_like(mean)
        sample = self.target._normal_sample(mean, std)
        assert not torch.allclose(sample, mean)
        assert sample.device == device

    def test_sample(self, device: torch.device) -> None:
        """Test MeanFieldNormalVarDist.sample."""
        vardist = self.target()
        mean = torch.randn((3, 4), device=device)
        log_std = torch.full_like(mean, -float("inf"), device=device)
        sample = vardist.sample(mean, log_std)
        assert sample.shape == mean.shape
        assert torch.allclose(sample, mean)
        assert sample.device == device

        mean = torch.randn((6,), device=device)
        log_std = torch.zeros_like(mean, device=device)
        sample = vardist.sample(mean, log_std)
        assert not torch.allclose(sample, mean)
        assert sample.device == device

    @pytest.mark.parametrize("norm_constants", [True, False])
    def test_normal_predictive_distribution(
        self, norm_constants: bool, device: torch.device
    ) -> None:
        """Test Normal as predicitve distribution."""
        predictive_dist = self.target()
        use_norm_constants(norm_constants)

        nr_samples = 50
        sample_shape = (5, 3)
        samples = torch.randn((nr_samples, *sample_shape), device=device)
        reference = torch.randn(sample_shape, device=device)
        target_mean = samples.mean(dim=0)
        target_std = samples.std(dim=0)
        target_log_prob = dist.Normal(target_mean, target_std).log_prob(reference)
        if not norm_constants:
            norm_const = (
                torch.full_like(target_mean, 2 * torch.pi, device=device).log() / 2
            )
            target_log_prob += norm_const

        test_mean, test_std = predictive_dist.predictive_parameters_from_samples(
            samples
        )
        assert torch.allclose(test_mean, target_mean)
        assert torch.allclose(test_std, target_std)
        assert test_mean.device == device
        assert test_std.device == device

        test_log_prob = predictive_dist.log_prob_from_parameters(
            reference, (target_mean, target_std)
        )
        assert torch.allclose(test_log_prob, target_log_prob, atol=1e-7)
        assert test_log_prob.device == device

        end2end_log_prob = predictive_dist.log_prob_from_samples(reference, samples)
        assert torch.allclose(end2end_log_prob, test_log_prob)
        assert end2end_log_prob.device == device


class TestMeanFieldNormal(TestNormal):
    """Test for Normal distribution alias."""

    target = MeanFieldNormal


ref_cholesky = torch.tensor([[0.8, 0.0, 0.0], [-0.1, 0.6, 0.0], [0.2, -0.5, 1.0]])
ref_variance = torch.matmul(ref_cholesky, ref_cholesky.mT)


class TestCorrelatedNormal:
    """Tests for correlated normal distribution."""

    target = CorrelatedNormal

    @pytest.mark.parametrize(
        "norm_constants,params",
        list(
            product(
                [True, False],
                [
                    None,
                    (0.7, 0.3, 0.1, None, None),
                    (0.0, 1.0, 0.0, None, ref_cholesky),
                    (-0.35, 1.0, 0.0, ref_variance, None),
                ],
            ),
        ),
    )
    def test_prior_log_prob(
        self,
        norm_constants: bool,
        params: Optional[
            Tuple[float, float, float, Optional[Tensor], Optional[Tensor]]
        ],
        device: torch.device,
    ) -> None:
        """Test CorrelatedNormal.prior_log_prob."""
        use_norm_constants(norm_constants)
        if params is not None:
            params = list(params)  # type:ignore[assignment]
            if params[3] is not None:
                params[3] = params[3].to(device)  # type:ignore[index]
            if params[4] is not None:
                params[4] = params[4].to(device)  # type:ignore[index]

        if params is None:
            prior = self.target()
            assert prior.mean == 0.0
            assert prior.log_std == 0.0
            assert prior.corr == 0.0
            assert prior.scale_tril is None
            assert prior.covariance_matrix is None
            params = (0.0, 1.0, 0.0, None, None)
        elif params[3] is None and params[4] is None:
            prior = self.target(*params)
            assert prior.mean == params[0]
            assert prior.log_std == log(params[1])
            assert prior.corr == params[2]
            assert prior.scale_tril is None
            assert prior.covariance_matrix is None
        elif params[3] is not None:
            prior = self.target(*params)
            assert prior.mean == params[0]
            assert prior.covariance_matrix is params[3]
            assert torch.allclose(prior.scale_tril, torch.linalg.cholesky(params[3]))
        else:  # param[4] is not None
            prior = self.target(*params)
            assert prior.mean == params[0]
            assert prior.scale_tril is params[4]
            assert torch.allclose(
                prior.covariance_matrix, torch.matmul(params[4], params[4].mT)
            )

        if params[3] is not None or params[4] is not None:
            shape: Tuple[int, ...] = (3,)
        else:
            shape = (5, 3)

        sample = torch.randn(*shape, device=device)

        vec_len = prod(shape)
        loc = torch.full([vec_len], params[0], device=device)

        covariance_matrix = params[3]
        scale_tril = params[4]

        if covariance_matrix is None and scale_tril is None:
            diag = torch.full([vec_len], params[1], device=device)
            scale_tril = torch.diagflat(diag)
            scale_tril[tuple(torch.tril_indices(vec_len, vec_len, -1))] = params[2]

        ref_dist = torch.distributions.MultivariateNormal(
            loc, covariance_matrix=covariance_matrix, scale_tril=scale_tril
        )

        # ref_samples = ref_dist.sample((10000,))
        # ref_mean = ref_samples.mean(dim=0).reshape(shape)
        # ref_corr = ref_samples.mT.cov()

        print("Ref: ", ref_dist.log_prob(sample.flatten()))
        print("Prior: ", prior.prior_log_prob(sample))
        assert torch.allclose(
            ref_dist.log_prob(sample.flatten()), prior.prior_log_prob(sample)
        )
