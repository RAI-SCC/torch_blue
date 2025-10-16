"""This module provides basic layers and loss functions for BNN-training with variational inference."""

from .analytical_kl_loss import (
    AnalyticalKullbackLeiblerLoss,
    KullbackLeiblerModule,
    NonBayesianDivergence,
    NormalNormalDivergence,
    UniformNormalDivergence,
)
from .base import VIModule
from .conv import VIConv1d, VIConv2d, VIConv3d
from .kl_loss import KullbackLeiblerLoss
from .linear import VILinear
from .sequential import VIResidualConnection, VISequential
from .transformer import (
    VIMultiheadAttention,
    VITransformer,
    VITransformerDecoder,
    VITransformerDecoderLayer,
    VITransformerEncoder,
    VITransformerEncoderLayer,
)
from .utils.common_types import VIkwargs
from .utils.vi_return import VIReturn

__all__ = [
    "AnalyticalKullbackLeiblerLoss",
    "KullbackLeiblerLoss",
    "VIConv1d",
    "VIConv2d",
    "VIConv3d",
    "VILinear",
    "VIModule",
    "VIMultiheadAttention",
    "VIResidualConnection",
    "VISequential",
    "VITransformer",
    "VITransformerDecoder",
    "VITransformerDecoderLayer",
    "VITransformerEncoder",
    "VITransformerEncoderLayer",
    "KullbackLeiblerModule",
    "NonBayesianDivergence",
    "NormalNormalDivergence",
    "UniformNormalDivergence",
    "VIkwargs",
    "VIReturn",
]
