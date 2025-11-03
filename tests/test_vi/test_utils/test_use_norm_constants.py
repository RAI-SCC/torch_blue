from torch_blue.vi import _globals
from torch_blue.vi.utils import use_norm_constants


def test_use_norm_constants() -> None:
    """Test setting of global variable _USE_NORM_CONSTANTS."""
    assert not _globals._USE_NORM_CONSTANTS
    use_norm_constants()
    assert _globals._USE_NORM_CONSTANTS
    use_norm_constants(False)
    assert not _globals._USE_NORM_CONSTANTS
