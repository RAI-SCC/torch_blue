"""Provides a collection of distributions."""

from .base import Distribution
from .categorical import Categorical
from .non_bayesian import NonBayesian, UniformPrior
from .normal import MeanFieldNormal
from .quiet import BasicQuietPrior
from .student_t import MeanFieldStudentT

Normal = MeanFieldNormal
StudentT = MeanFieldStudentT

__all__ = [
    "BasicQuietPrior",
    "Categorical",
    "Distribution",
    "MeanFieldNormal",
    "MeanFieldStudentT",
    "NonBayesian",
    "Normal",
    "StudentT",
    "UniformPrior",
]
