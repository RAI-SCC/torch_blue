from typing import Any, Callable

import pytest
import torch

from torch_bayesian import vi
from torch_bayesian.vi import VIModule, convert_to_vimodule


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Detect and return local device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def get_model() -> Callable[..., VIModule]:
    """Create module getter."""

    def getter(model_name: str, *args: Any, **kwargs: Any) -> VIModule:
        if hasattr(vi, model_name):
            return getattr(vi, model_name)(*args, **kwargs)
        elif hasattr(torch.nn, model_name):
            module = getattr(torch.nn, model_name)(*args, **kwargs)
            convert_to_vimodule(module, *args, **kwargs)
            return module
        else:
            raise ValueError(f"{model_name} is not a valid module")

    return getter
