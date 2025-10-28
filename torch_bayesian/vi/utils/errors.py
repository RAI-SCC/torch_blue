class NoVariablesError(Exception):
    """
    Raised by VIModules when an operation requires nonexistent variables.

    If a VIModule does not contain random variables (i.e. weight matrices), some methods
    that work only with or on them raise this error, since they should not be called.
    """

    pass


class UnsupportedDistributionError(ValueError):
    r"""
    Raised if a distribution is used in an unsupported way.

    A :class:`~torch\_bayesian.vi.distributions.Distribution` may not support being
    used as prior, variational distribution or predictive distribution. When attempting
    to still use it in such a way this error is raised.
    """
