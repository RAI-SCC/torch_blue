"""Provides a collection of distributions."""

from .base import Distribution
from .non_bayesian import NonBayesian, UniformPrior
from .normal import MeanFieldNormal

Normal = MeanFieldNormal

__all__ = [
    "Distribution",
    "MeanFieldNormal",
    "NonBayesian",
    "Normal",
    "UniformPrior",
]
