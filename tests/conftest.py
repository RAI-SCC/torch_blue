import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Detect and return local device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
