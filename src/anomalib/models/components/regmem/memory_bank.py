"""Patch memory bank construction utilities with GraphCore support."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

def _ensure_same_device(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Ensure all tensors are on the same device."""
    if not tensors:
        return tensors
    
    target_device = tensors[0].device
    return tuple(tensor.to(target_device) if tensor.device != target_device else tensor for tensor in tensors)


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
    """Build a compact memory bank of registered patch descriptors with GraphCore support."""

    def __init__(
        self,
        layers: list[str],
        coreset_sampling_ratio: float = 0.05,
        device: torch.device | None = None,
        use_graphcore: bool = True,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.device = device
        self.use_graphcore = use_graphcore
        
        # Add storage for GraphCore features
        self.reset()

    def reset(self) -> None:
        self.register_buffer("_is_built", torch.tensor(False), persistent=True)
        self._storage: dict[str, list[Tensor]] = {layer: [] for layer in self.layers}
        self._weights: dict[str, list[Tensor]] = {layer: [] for layer in self.layers}
        self._graphcore_storage: list[Tensor] = []  # Store GraphCore features separately
        self.memory: dict[str, MemoryBankItem] = {}
        self.graphcore_memory: MemoryBankItem | None = None

    def update(self, features: Mapping[str, Tensor], weights: Mapping[str, Tensor] | None = None, 
               graphcore_features: Tensor | None = None) -> None:
        """Accumulate support features prior to coreset sampling."""

        weights = weights or {}
        for layer in self.layers:
            feats = _flatten_patches(features[layer].detach())
            device = self.device or feats.device
            feats = feats.to(device)
            weight = weights.get(layer)
            if weight is not None:
                weight = weight.reshape(-1).to(device)
            self._storage[layer].append(feats)
            if weight is not None:
                self._weights[layer].append(weight)
                
        # Store GraphCore features if available
        if self.use_graphcore and graphcore_features is not None:
            device = self.device or graphcore_features.device
            self._graphcore_storage.append(graphcore_features.detach().to(device))

    def build(self) -> None:
        """Apply greedy coreset sampling and finalise the memory bank."""

        self.memory.clear()
        self.graphcore_memory = None
        
        # Build memory for each layer
        for layer in self.layers:
            if not self._storage[layer]:
                continue
            device = self.device or self._storage[layer][0].device
            # Ensure all tensors are on the same device before concatenation
            storage_tensors = [tensor.to(device) for tensor in self._storage[layer]]
            # Double-check device synchronization
            storage_tensors = list(_ensure_same_device(*storage_tensors))
            features = torch.cat(storage_tensors, dim=0)
            if self._weights[layer]:
                weight_tensors = [tensor.to(device) for tensor in self._weights[layer]]
                weights = torch.cat(weight_tensors, dim=0)
            else:
                weights = torch.ones(features.shape[0], device=device)
            num_samples = max(1, int(features.shape[0] * self.coreset_sampling_ratio))
            sampled, indices = greedy_coreset(features, num_samples=num_samples)
            sampled_weights = _normalize_weights(weights[indices], sampled.shape[0], device=device)
            self.memory[layer] = MemoryBankItem(features=sampled, weights=sampled_weights)

        # Build GraphCore memory if available
        if self.use_graphcore and self._graphcore_storage:
            device = self.device or self._graphcore_storage[0].device
            # Ensure all tensors are on the same device before concatenation
            graphcore_tensors = [tensor.to(device) for tensor in self._graphcore_storage]
            graphcore_features = torch.cat(graphcore_tensors, dim=0)
            num_samples = max(1, int(graphcore_features.shape[0] * self.coreset_sampling_ratio))
            sampled, indices = greedy_coreset(graphcore_features, num_samples=num_samples)
            weights = torch.ones(sampled.shape[0], device=device)
            self.graphcore_memory = MemoryBankItem(features=sampled, weights=weights)

        if self.memory or self.graphcore_memory:
            device = self.device or torch.device("cpu")
        else:
            device = self.device or torch.device("cpu")
        self._is_built = torch.tensor(True, device=device)

    def __len__(self) -> int:
        return sum(item.features.shape[0] for item in self.memory.values())

    def get(self, layer: str) -> MemoryBankItem:
        return self.memory[layer]

