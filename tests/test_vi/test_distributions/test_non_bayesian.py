from typing import Optional, Tuple, Union

import pytest
import torch
from torch import nn

from torchbuq.vi.distributions import NonBayesian, UniformPrior

shapes = [5, [3, 6], [5, 3, 4]]


class TestNonBayesian:
    """Tests non-Bayesian distribution."""

    target = NonBayesian

    def test_flags(self) -> None:
        """Test correct labeling."""
        assert self.target.is_prior
        assert self.target.is_variational_distribution
        assert self.target.is_predictive_distribution

    @pytest.mark.parametrize("shape", shapes)
    def test_prior_log_prob(
        self, shape: Union[int, Tuple[int, ...]], device: torch.device
    ) -> None:
        """Test prior log probability."""
        dist = self.target()
        sample = torch.randn(shape)

        out = dist.prior_log_prob(sample)
        assert out == 0.0
        assert out.device == device

    @pytest.mark.parametrize("shape", shapes)
    def test_variational_log_prob(
        self, shape: Union[int, Tuple[int, ...]], device: torch.device
    ) -> None:
        """Test variational log probability."""
        dist = self.target()
        mean = torch.randn(shape, device=device)
        sample = dist.sample(mean)
        ref = torch.zeros_like(mean, device=device)

        log_prob = dist.variational_log_prob(sample, mean)
        assert torch.allclose(ref, log_prob)
        assert log_prob.device == device

    @pytest.mark.parametrize("shape", shapes)
    def test_sample(
        self, shape: Union[int, Tuple[int, ...]], device: torch.device
    ) -> None:
        """Test sample()."""
        dist = self.target()
        mean = torch.randn(shape, device=device)
        sample = dist.sample(mean)
        assert sample.shape == mean.shape
        assert torch.allclose(sample, mean)
        assert sample.device == device

    @pytest.mark.parametrize(
        "loss_type, target",
        [
            ("MSE", nn.MSELoss),
            ("L2", nn.MSELoss),
            ("MAE", nn.L1Loss),
            ("L1", nn.L1Loss),
            ("Error", None),
            (None, "Error"),
        ],
    )
    def test_non_bayesian_predictive_distribution(
        self,
        loss_type: str,
        target: Optional[Union[nn.Module, str]],
        device: torch.device,
    ) -> None:
        """Test Non-Bayesian Predictive Distribution."""
        if target is None:
            with pytest.raises(ValueError, match=f"Unsupported loss type: {loss_type}"):
                _ = self.target(loss_type)
            return

        predictive_distribution = self.target(loss_type)
        if type(target) is nn.Module:
            assert isinstance(predictive_distribution.loss, target)

        sample_shape = (4, 7)
        samples = torch.randn((1, *sample_shape), device=device)
        reference = torch.randn(sample_shape, device=device)

        predictive_mean = predictive_distribution.predictive_parameters_from_samples(
            samples
        )
        try:
            loss = predictive_distribution.log_prob_from_samples(reference, samples)
        except ValueError as e:
            if str(e) == "loss_type must be set during initialization":
                return
            raise

        target_mean = samples.mean(dim=0)
        target_loss = target()(target_mean, reference)

        assert target_mean.shape == predictive_mean.shape
        assert predictive_mean.device == device
        assert torch.allclose(target_mean, predictive_mean)

        assert target_loss.shape == loss.shape
        assert loss.device == device
        assert torch.allclose(target_loss, loss)


class TestUniformPrior(TestNonBayesian):
    """Tests Uniform Prior."""

    target = UniformPrior

    def test_flags(self) -> None:
        """Test correct labeling."""
        assert self.target.is_prior
        assert not self.target.is_variational_distribution
        assert not self.target.is_predictive_distribution
