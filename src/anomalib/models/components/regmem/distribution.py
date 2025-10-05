"""Distribution estimators for registered patch features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MahalanobisStatistics:
    """Container storing statistics required for Mahalanobis distances."""

    mean: Tensor
    covariance_inv: Tensor


def _compute_mahalanobis_stats(features: Tensor, eps: float = 1e-6) -> MahalanobisStatistics:
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    covariance = centered.T @ centered / max(1, features.shape[0] - 1)
    covariance = covariance + eps * torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    covariance_inv = torch.linalg.pinv(covariance)
    return MahalanobisStatistics(mean=mean, covariance_inv=covariance_inv)


def _mahalanobis_distance(features: Tensor, stats: MahalanobisStatistics) -> Tensor:
    centered = features - stats.mean
    left = centered @ stats.covariance_inv
    return (left * centered).sum(dim=1)


def _normalize(features: Tensor) -> Tensor:
    return torch.nn.functional.normalize(features, dim=1)


def _gaussian_kernel(x: Tensor, y: Tensor, sigma: float = 1.0) -> Tensor:
    return torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma**2))


class DistributionEstimator(nn.Module):
    """Hybrid distribution estimator combining multiple distance metrics."""

    def __init__(
        self,
        combine_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        mmd_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.combine_weights = combine_weights
        self.mmd_sigma = mmd_sigma
        self.stats: dict[str, MahalanobisStatistics] = {}
        self.stats_pp: dict[str, MahalanobisStatistics] = {}
        self.memory_features: dict[str, Tensor] = {}

    def fit(self, memory: dict[str, Tensor]) -> None:
        self.stats.clear()
        self.stats_pp.clear()
        self.memory_features = {layer: feats for layer, feats in memory.items()}
        for layer, feats in memory.items():
            self.stats[layer] = _compute_mahalanobis_stats(feats)
            self.stats_pp[layer] = _compute_mahalanobis_stats(_normalize(feats))

    def _mahalanobis(self, layer: str, features: Tensor) -> Tensor:
        return _mahalanobis_distance(features, self.stats[layer])

    def _mahalanobis_pp(self, layer: str, features: Tensor) -> Tensor:
        return _mahalanobis_distance(_normalize(features), self.stats_pp[layer])

    def _mmd(self, layer: str, features: Tensor) -> Tensor:
        memory = self.memory_features[layer]
        kernel_xx = _gaussian_kernel(features, features, sigma=self.mmd_sigma)
        kernel_yy = _gaussian_kernel(memory, memory, sigma=self.mmd_sigma)
        kernel_xy = _gaussian_kernel(features, memory, sigma=self.mmd_sigma)
        mmd = kernel_xx.mean(dim=1) + kernel_yy.mean() - 2 * kernel_xy.mean(dim=1)
        return mmd

    def score(self, layer: str, features: Tensor) -> Tensor:
        m = self._mahalanobis(layer, features)
        mpp = self._mahalanobis_pp(layer, features)
        mmd = self._mmd(layer, features)
        w1, w2, w3 = self.combine_weights
        return w1 * m + w2 * mpp + w3 * mmd

