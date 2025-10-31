import torch

from torch_blue.vi.distributions import Categorical


def test_categorical(device: torch.device) -> None:
    """Test Categorical."""
    predictive_dist = Categorical()
    predictive_prob = Categorical(input_type="probs")

    samples = 4
    batch = 3
    categories = 5
    probs = torch.rand((samples, batch, categories), device=device)
    target = torch.randint(0, categories, (batch,), device=device)

    p1 = predictive_dist.predictive_parameters_from_samples(probs.log())
    p2 = predictive_prob.predictive_parameters_from_samples(probs)
    assert p1.shape == (batch, categories)
    assert p2.shape == (batch, categories)
    assert torch.allclose(p1, p2)
    assert p1.device == device
    assert p2.device == device

    ref_dist1 = torch.distributions.Categorical(probs=probs)
    assert torch.allclose(p1, ref_dist1.probs.mean(dim=0))

    ref_dist2 = torch.distributions.Categorical(
        probs=ref_dist1.probs.mean(dim=0), validate_args=False
    )
    ref_dist2.probs = ref_dist2.probs + 1e-5
    target_log_prob1 = ref_dist2.log_prob(target)

    ref_dist3 = torch.distributions.Categorical(
        probs=ref_dist1.probs.mean(dim=0), validate_args=False
    )
    target_log_prob2 = ref_dist3.log_prob(target)

    log_prob1 = predictive_dist.log_prob_from_parameters(target, p1)
    log_prob2 = predictive_dist.log_prob_from_parameters(target, p2, eps=0.0)
    assert log_prob1.shape == (batch,)
    assert torch.allclose(log_prob1, target_log_prob1)
    assert torch.allclose(log_prob2, target_log_prob2)
    assert log_prob1.device == device
    assert log_prob2.device == device
