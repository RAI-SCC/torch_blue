class NoVariablesError(Exception):
    """
    Raised by VIModules when an operation requires nonexistent variables.

    If a VIModule does not contain random variables (i.e. weight matrices), some methods
    that work only with or on them raise this error, since they should not be called.
    """

    pass
