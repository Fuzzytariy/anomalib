"""Model components for various anomaly detection architectures."""

from .backbone import get_decoder
from .base import AnomalibModule, BufferListMixin, DynamicBufferMixin, MemoryBankMixin
from .classification import KDEClassifier
from .cluster import GaussianMixture, KMeans
from .dimensionality_reduction import PCA, SparseRandomProjection
from .feature_extractors import TimmFeatureExtractor
from .filters import GaussianBlur2d
from .flow import AllInOneBlock
from .graphcore import GraphCore
from .layers import SSPCAB
from .regmem import (
    DistributionEstimator,
    FeatureRegistrationModule,
    MahalanobisStatistics,
    MemoryBankItem,
    PatchMemoryBank,
    RegistrationBlock,
    RegistrationOutputs,
    greedy_coreset,
)
from .sampling import KCenterGreedy
from .stats import GaussianKDE, MultiVariateGaussian

__all__ = [
    "AllInOneBlock",
    "AnomalibModule",
    "BufferListMixin",
    "DynamicBufferMixin",
    "GaussianBlur2d",
    "GaussianMixture",
    "GaussianKDE",
    "GraphCore",
    "KDEClassifier",
    "KMeans",
    "KCenterGreedy",
    "MahalanobisStatistics",
    "MemoryBankItem",
    "MemoryBankMixin",
    "MultiVariateGaussian",
    "PCA",
    "PatchMemoryBank",
    "SparseRandomProjection",
    "SSPCAB",
    "TimmFeatureExtractor",
    "DistributionEstimator",
    "FeatureRegistrationModule",
    "RegistrationBlock",
    "RegistrationOutputs",
    "get_decoder",
    "greedy_coreset",

]
