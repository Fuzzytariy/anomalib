"""Feature registration and memory components for few-shot anomaly detection.

This package bundles reusable building blocks that are shared across the
few-shot registration-memory (RegMem) pipeline:

* :mod:`registration` - differentiable feature registration networks that
  align support and query representations.
* :mod:`memory_bank` - utilities to construct a compact memory bank of normal
  patch descriptors with greedy coreset sampling.
* :mod:`distribution` - distribution estimators that mix Mahalanobis distance,
  Mahalanobis++ normalisation and maximum mean discrepancy (MMD) metrics.

The components are implemented as lightweight, well documented PyTorch modules
so that they can be reused in other research prototypes without depending on
the high level Lightning interface.
"""

from .distribution import DistributionEstimator, MahalanobisStatistics
from .memory_bank import PatchMemoryBank
from .registration import FeatureRegistrationModule, RegistrationOutputs

__all__ = [
    "DistributionEstimator",
    "FeatureRegistrationModule",
    "MahalanobisStatistics",
    "PatchMemoryBank",
    "RegistrationOutputs",
]

