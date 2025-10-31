from collections import OrderedDict
from typing import Tuple, Union

import pytest
import torch
from torch import Tensor
from torch.nn import ReLU

from torchbuq.vi import VILinear, VIModule, VIResidualConnection, VISequential


def test_sequential(device: torch.device) -> None:
    """Test VISequential."""
    in_features = 2
    hidden_features = 4
    out_features = 3

    broke_module_dict = OrderedDict(
        module1=VILinear(
            in_features, hidden_features, return_log_probs=False, device=device
        ),
        activation=ReLU(),
        module2=VILinear(hidden_features, out_features, device=device),
    )

    with pytest.raises(AssertionError, match="return_log_probs *"):
        _ = VISequential(broke_module_dict)

    module_dict = OrderedDict(
        module1=VILinear(in_features, hidden_features, device=device),
        activation=ReLU(),
        module2=VILinear(hidden_features, out_features, device=device),
    )

    model1 = VISequential(module_dict)
    assert model1._return_log_probs

    module_list = list(module_dict.values())
    module_list[0].return_log_probs = False
    module_list[2].return_log_probs = False

    model2 = VISequential(*module_list)
    assert not model2._return_log_probs

    for m1, m2 in zip(model1.modules(), model2.modules()):
        if isinstance(m1, VISequential) or isinstance(m1, ReLU):
            continue
        assert torch.allclose(m1._weight_mean, m2._weight_mean)
        assert m1._weight_mean.device == device
        assert m2._weight_mean.device == device

    sample = torch.randn(2, in_features, device=device)

    model1.return_log_probs = True
    model2.return_log_probs = True
    out1 = model1(sample, samples=5)
    out2 = model2(sample, samples=5)
    assert out1.shape == (5, 2, out_features)
    assert out1.device == device
    assert out2.shape == (5, 2, out_features)
    assert out2.device == device

    model1.return_log_probs = False
    model2.return_log_probs = False
    out1 = model1(sample, samples=4)
    out2 = model2(sample, samples=4)
    assert out1.shape == (4, 2, out_features)
    assert out1.device == device
    assert out2.shape == (4, 2, out_features)
    assert out2.device == device

    module3 = VISequential(ReLU(), ReLU())
    assert not module3._return_log_probs


def test_residual_connection(device: torch.device) -> None:
    """Test VIResidualConnection."""

    class Test(VIModule):
        def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
            if self._return_log_probs:
                self._log_probs = dict(
                    all=[torch.tensor([[0.0, 1.0]], device=x.device)]
                )
            return x

    class Test2(VIModule):
        def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
            if self._return_log_probs:
                self._log_probs = dict(
                    all=[torch.tensor([[0.0, 1.0]], device=x.device)]
                )
            return x.reshape((2, 9))

    module = VIResidualConnection(Test())
    broken_module = VIResidualConnection(Test2())
    module.return_log_probs = True
    sample1 = torch.randn(6, 3, device=device)
    out1 = module(sample1, samples=3)
    lps1 = out1.log_probs
    plp1 = lps1[:, 0]
    vlp1 = lps1[:, 1]
    with pytest.raises(
        RuntimeError,
        match="Output shape \(torch.Size\(\[2, 9\]\)\) of residual connection must match input shape \(torch.Size\(\[6, 3\]\)\)",
    ):
        broken_module(sample1, samples=3)

    assert torch.allclose(out1.mean(0), 2 * sample1)
    assert torch.allclose(plp1, torch.zeros_like(plp1))
    assert torch.allclose(vlp1, torch.ones_like(vlp1))
    assert out1.device == device
    assert plp1.device == device
    assert vlp1.device == device

    module.return_log_probs = False
    broken_module.return_log_probs = False
    sample2 = torch.randn(3, 6, device=device)
    out2 = module(sample2, samples=5)
    assert out2.device == device
    assert torch.allclose(out2.mean(0), 2 * sample2)
    with pytest.raises(
        RuntimeError,
        match="Output shape \(torch.Size\(\[2, 9\]\)\) of residual connection must match input shape \(torch.Size\(\[3, 6\]\)\)",
    ):
        broken_module(sample2, samples=5)
