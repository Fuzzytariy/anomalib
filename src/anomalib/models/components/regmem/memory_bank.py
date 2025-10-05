"""Patch memory bank construction utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


def _flatten_patches(features: Tensor) -> Tensor:
    """Flatten spatial dimensions to create patch descriptors."""

    batch, channels, height, width = features.shape
    return features.permute(0, 2, 3, 1).reshape(batch * height * width, channels)


def _normalize_weights(weights: Tensor | None, num_items: int, device: torch.device) -> Tensor:
    if weights is None:
        return torch.ones(num_items, device=device)
    if weights.ndim == 0:
        return torch.full((num_items,), float(weights), device=device)
    if weights.numel() != num_items:
        msg = "Number of weights must match number of items."
        raise ValueError(msg)
    return weights.to(device=device)


def greedy_coreset(features: Tensor, num_samples: int) -> tuple[Tensor, torch.Tensor]:
    """Greedy coreset selection used in PatchCore.

    Returns both sampled features and the indices of the selected samples.
    """

    if num_samples >= features.shape[0]:
        indices = torch.arange(features.shape[0], device=features.device)
        return features, indices

    remaining = features
    selected_indices: list[int] = []
    idx = torch.randint(0, remaining.shape[0], (1,), device=features.device).item()
    selected_indices.append(idx)
    selected = remaining[idx].unsqueeze(0)
    min_distances = torch.full((remaining.shape[0],), float("inf"), device=features.device)

    for _ in range(1, num_samples):
        distances = torch.cdist(selected[-1].unsqueeze(0), remaining, p=2).squeeze(0)
        min_distances = torch.minimum(min_distances, distances)
        next_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(next_idx)
        selected = torch.cat([selected, remaining[next_idx].unsqueeze(0)], dim=0)

    indices = torch.tensor(selected_indices, device=features.device)
    return features[indices], indices


@dataclass
class MemoryBankItem:
    """Store sampled features and optional weights for a single layer."""

    features: Tensor
    weights: Tensor


class PatchMemoryBank(nn.Module):
    """Build a compact memory bank of registered patch descriptors."""

    def __init__(
        self,
        layers: list[str],
        coreset_sampling_ratio: float = 0.05,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.register_buffer("_is_built", torch.tensor(False), persistent=True)
        self._storage: dict[str, list[Tensor]] = {layer: [] for layer in self.layers}
        self._weights: dict[str, list[Tensor]] = {layer: [] for layer in self.layers}
        self.memory: dict[str, MemoryBankItem] = {}

    def update(self, features: Mapping[str, Tensor], weights: Mapping[str, Tensor] | None = None) -> None:
        """Accumulate support features prior to coreset sampling."""

        weights = weights or {}
        for layer in self.layers:
            feats = _flatten_patches(features[layer].detach())
            weight = weights.get(layer)
            if weight is not None:
                weight = weight.reshape(-1)
            self._storage[layer].append(feats.cpu())
            if weight is not None:
                self._weights[layer].append(weight.cpu())

    def build(self) -> None:
        """Apply greedy coreset sampling and finalise the memory bank."""

        device = self.device or torch.device("cpu")
        self.memory.clear()
        for layer in self.layers:
            if not self._storage[layer]:
                continue
            features = torch.cat(self._storage[layer], dim=0).to(device)
            if self._weights[layer]:
                weights = torch.cat(self._weights[layer], dim=0).to(device)
            else:
                weights = torch.ones(features.shape[0], device=device)
            num_samples = max(1, int(features.shape[0] * self.coreset_sampling_ratio))
            sampled, indices = greedy_coreset(features, num_samples=num_samples)
            sampled_weights = _normalize_weights(weights[indices], sampled.shape[0], device=device)
            self.memory[layer] = MemoryBankItem(features=sampled, weights=sampled_weights)

        self._is_built = torch.tensor(True, device=device)

    def __len__(self) -> int:
        return sum(item.features.shape[0] for item in self.memory.values())

    def get(self, layer: str) -> MemoryBankItem:
        return self.memory[layer]

