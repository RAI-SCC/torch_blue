from warnings import filterwarnings

import torch
from pytest import warns

from torch_bayesian.vi import KullbackLeiblerLoss, VIReturn
from torch_bayesian.vi.distributions import MeanFieldNormal


def test_kl_loss(device: torch.device) -> None:
    """Test Kullback-LeiblerLoss."""
    sample_nr = 8
    batch_size = 4
    sample_shape = (5, 3)
    samples = torch.randn((sample_nr, batch_size, *sample_shape), device=device)
    log_probs = torch.randn((batch_size, 2), device=device)
    target = torch.randn([batch_size, *sample_shape], device=device)

    model_return = VIReturn(samples, log_probs)
    double_return = VIReturn(torch.cat([samples] * 2, dim=1), log_probs)
    double_target = VIReturn(torch.cat([target] * 2, dim=0), None)

    loss1 = KullbackLeiblerLoss(MeanFieldNormal())
    with warns(
        UserWarning,
        match=f"No dataset_size is provided. Batch size \({batch_size}\) is used instead.",
    ):
        _ = loss1(model_return, target)

    out1 = loss1(model_return, target, dataset_size=batch_size)
    ref_data_fit = (
        -batch_size
        * loss1.predictive_distribution.log_prob_from_samples(target, samples)
        .mean(0)
        .sum()
    )
    ref_kl_term = log_probs.mean(0)[1] - log_probs.mean(0)[0]
    assert out1 == (ref_data_fit + ref_kl_term)
    assert out1.shape == ()
    assert out1.device == device
    assert loss1.log is None
    assert not loss1._track

    loss2 = KullbackLeiblerLoss(MeanFieldNormal(), dataset_size=batch_size)
    out2 = loss2(model_return, target)
    assert out1 == out2
    assert out2.device == device
    out3 = loss2(model_return, target, dataset_size=2 * batch_size)
    assert out1 != out3
    assert out3.device == device

    loss3 = KullbackLeiblerLoss(MeanFieldNormal(), dataset_size=2 * batch_size)
    out4 = loss3(model_return, target)
    assert out1 != out4
    assert out3 == out4
    assert out4.device == device

    loss4 = KullbackLeiblerLoss(MeanFieldNormal(), dataset_size=batch_size, heat=0.5)
    out5 = loss4(model_return, target)
    assert out1 != out5
    assert out5.device == device

    filterwarnings("ignore", category=UserWarning)
    out6 = loss1(model_return, target)
    assert out1 == out6
    assert out6.device == device

    loss5 = KullbackLeiblerLoss(MeanFieldNormal(), dataset_size=batch_size, track=True)

    assert loss5.log is not None
    assert loss5._track

    loss1.track()
    assert loss1.log is not None
    assert loss1._track

    for key in ["data_fitting", "prior_matching", "log_probs"]:
        assert loss1.log[key] == []
        assert loss5.log[key] == []

    loss1(model_return, target, dataset_size=batch_size)
    loss5(model_return, target, dataset_size=batch_size)

    for key in ["data_fitting", "prior_matching", "log_probs"]:
        assert len(loss1.log[key]) == 1
        assert len(loss5.log[key]) == 1
        comp = loss1.log[key][0] == loss5.log[key][0]
        if isinstance(comp, bool):
            assert comp
        else:
            assert comp.all()

    assert loss1.log["log_probs"][0][1] - loss1.log["log_probs"][0][0] == ref_kl_term
    assert loss1.log["data_fitting"][0] + loss1.log["prior_matching"][0] == out1

    double_out = loss1(double_return, double_target, dataset_size=batch_size)
    assert torch.allclose(double_out, out1)
    assert double_out.device == device
