"""Provides a collection of distributions."""

from .base import Distribution
from .normal import MeanFieldNormal

Normal = MeanFieldNormal

__all__ = [
    "Distribution",
    "MeanFieldNormal",
    "Normal",
]
