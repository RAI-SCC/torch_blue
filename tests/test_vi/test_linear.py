from math import exp
from typing import cast

import torch
from torch import Tensor

from torchbuq.vi import VILinear, VIReturn
from torchbuq.vi.distributions import MeanFieldNormal, NonBayesian


def test_vilinear(device: torch.device) -> None:
    """Test VILinear and prior initialization."""
    in_features = 3
    out_features = 5
    module1 = VILinear(in_features, out_features, return_log_probs=False, device=device)
    assert module1.in_features == in_features
    assert module1.out_features == out_features
    assert module1.random_variables == ("weight", "bias")

    sample1 = torch.randn(4, in_features, device=device)
    out = module1(sample1, samples=10)
    assert out.shape == (10, 4, out_features)
    out.sum().backward()

    # test gradient presence
    for attr in ["_weight_mean", "_bias_mean", "_weight_log_std", "_bias_log_std"]:
        assert hasattr(module1, attr)
        assert getattr(module1, attr).requires_grad
        assert getattr(module1, attr).grad is not None

    module1._has_sampling_responsibility = False
    out = module1(sample1)
    assert out.shape == (4, out_features)

    sample2 = torch.randn(3, 2, in_features, device=device)
    multisample: Tensor = module1.sampled_forward(sample2, samples=4)
    assert multisample.shape == (4, 3, 2, out_features)
    for s in multisample[1:]:
        assert not (s == multisample[0]).all()
    multisample.sum().backward()

    # test bias == False
    in_features = 4
    out_features = 3
    module2 = VILinear(
        in_features, out_features, bias=False, return_log_probs=False, device=device
    )
    assert module2.random_variables == ("weight", "bias")
    assert not hasattr(module2, "_bias_mean")

    sample2 = torch.randn(6, in_features, device=device)
    out = module2(sample2, samples=4)
    assert out.shape == (4, 6, out_features)

    # test return_log_probs == True
    in_features = 2
    out_features = 5
    module3 = VILinear(in_features, out_features, return_log_probs=True, device=device)

    sample3 = torch.randn(4, 7, in_features, device=device)
    out = module3(sample3, samples=5)
    lps = out.log_probs
    assert out.shape == (5, 4, 7, out_features)
    assert lps.shape == (5, 2)

    multisample2 = cast(VIReturn, module3.sampled_forward(sample3, samples=10))
    multilps = cast(Tensor, multisample2.log_probs)
    assert multisample2.shape == (10, 4, 7, out_features)
    assert multilps.shape == (10, 2)
    multisample2.sum().backward()

    # test prior_init
    in_features = 7
    out_features = 3
    module4 = VILinear(
        in_features,
        out_features,
        prior_initialization=True,
        return_log_probs=False,
        device=device,
    )

    weight_mean = module4._weight_mean.clone()
    weight_log_std = module4._weight_log_std.clone()
    bias_mean = module4._bias_mean.clone()
    bias_log_std = module4._bias_log_std.clone()

    assert (weight_mean == torch.zeros_like(weight_mean)).all()
    assert (weight_log_std == torch.zeros_like(weight_log_std)).all()
    assert (bias_mean == torch.zeros_like(bias_mean)).all()
    assert (bias_log_std == torch.zeros_like(bias_log_std)).all()

    module4.reset_variational_parameters()
    assert (weight_mean == module4._weight_mean).all()
    assert (weight_log_std == module4._weight_log_std).all()
    assert (bias_mean == module4._bias_mean).all()
    assert (bias_log_std == module4._bias_log_std).all()

    in_features = 6
    out_features = 5
    prior = MeanFieldNormal(mean=1.0, std=exp(5.0))
    module5 = VILinear(
        in_features,
        out_features,
        prior=prior,
        prior_initialization=True,
        return_log_probs=False,
        device=device,
    )

    weight_mean = module5._weight_mean.clone()
    weight_log_std = module5._weight_log_std.clone()
    bias_mean = module5._bias_mean.clone()
    bias_log_std = module5._bias_log_std.clone()

    assert (weight_mean == torch.ones_like(weight_mean)).all()
    assert (weight_log_std == torch.full_like(weight_log_std, 5.0)).all()
    assert (bias_mean == torch.ones_like(bias_mean)).all()
    assert (bias_log_std == torch.full_like(bias_log_std, 5.0)).all()

    module6 = VILinear(
        in_features,
        out_features,
        variational_distribution=NonBayesian(),
        return_log_probs=False,
        device=device,
    )
    sample4 = torch.randn(4, in_features, device=device)
    out = module6(sample4, samples=1)
    assert out.shape == (1, 4, out_features)
