from torch_bayesian.vi import _globals


def use_norm_constants(mode: bool = True) -> None:
    """
    Set global flag _USE_NORM_CONSTANTS.

    This flag makes all distributions add normalization constants during log_prob
    calculation. These constants are mathematically accurate, but not needed and
    seemingly counterproductive for training, possibly due to float accuracy.

    Parameters
    ----------
    mode: bool, default: True
        Value to set the global flag to.
    """
    _globals._USE_NORM_CONSTANTS = mode
