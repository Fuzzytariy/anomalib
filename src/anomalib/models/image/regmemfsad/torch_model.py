"""Low level PyTorch module for the RegMem few-shot anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn

from anomalib.models.components.regmem import (
    DistributionEstimator,
    FeatureRegistrationModule,
    PatchMemoryBank,
)


@dataclass
class AnomalyPredictions:
    """Container storing anomaly scores and maps."""

    pred_score: Tensor
    anomaly_map: Tensor
    per_layer_maps: dict[str, Tensor]


class RegMemFewShotModel(nn.Module):
    """Encapsulates registration, memory bank construction and scoring."""

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.05,
        num_neighbors: int = 5,
        distribution_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        mmd_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.num_neighbors = num_neighbors
        self.registration = FeatureRegistrationModule(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
        )
        self.memory_bank = PatchMemoryBank(layers=self.layers, coreset_sampling_ratio=coreset_sampling_ratio)
        self.distribution_estimator = DistributionEstimator(
            combine_weights=distribution_weights, mmd_sigma=mmd_sigma
        )

    def reset(self) -> None:
        self.memory_bank.reset()

    def collect_support_features(self, images: Tensor) -> Tensor:
        self.memory_bank.device = images.device
        features = self.registration.extract(images)
        registered = self.registration.register_support_batch(features)
        self.memory_bank.update(registered)
        return torch.tensor(0.0, device=images.device)

    def build_memory_bank(self) -> None:
        self.memory_bank.build()
        if not self.memory_bank.memory:
            msg = "Memory bank is empty. Ensure that support features are collected before calling build_memory_bank()."
            raise RuntimeError(msg)
        memory = {layer: self.memory_bank.get(layer).features for layer in self.memory_bank.memory}
        self.distribution_estimator.fit(memory)

    def _score_layer(self, layer: str, query: Tensor) -> Tensor:
        if layer not in self.memory_bank.memory:
            msg = f"Requested layer {layer} not present in memory bank."
            raise KeyError(msg)
        item = self.memory_bank.get(layer)
        query_flat = query.permute(0, 2, 3, 1).reshape(-1, query.shape[1])
        distances = torch.cdist(query_flat, item.features, p=2)
        knn = torch.topk(distances, k=min(self.num_neighbors, item.features.shape[0]), dim=1, largest=False)
        knn_score = knn.values.mean(dim=1)
        dist_score = self.distribution_estimator.score(layer, query_flat)
        combined = 0.5 * knn_score + 0.5 * dist_score
        return combined

    def predict(self, images: Tensor) -> AnomalyPredictions:
        features = self.registration.extract(images)
        aligned_query = self.registration.align_features_to_prototypes(features)
        per_layer_maps: dict[str, Tensor] = {}
        layer_maps: list[Tensor] = []
        for layer in self.layers:
            query_features = aligned_query[layer]
            scores = self._score_layer(layer, query_features)
            batch, _, height, width = query_features.shape
            anomaly_map = scores.view(batch, height, width)
            per_layer_maps[layer] = anomaly_map
            layer_maps.append(anomaly_map)

        stacked_maps = torch.stack(layer_maps, dim=0)
        combined_map = stacked_maps.mean(dim=0)
        flattened = combined_map.view(combined_map.shape[0], -1)
        if flattened.shape[1] == 0:
            final_scores = torch.zeros(flattened.shape[0], device=flattened.device)
        else:
            topk = min(10, flattened.shape[1])
            final_scores = flattened.topk(k=topk, dim=1).values.mean(dim=1)
        return AnomalyPredictions(pred_score=final_scores, anomaly_map=combined_map, per_layer_maps=per_layer_maps)

    def registration_loss(self, images: Tensor) -> Tensor:
        if images.shape[0] < 2:
            # Return a differentiable zero so that Lightning's backward pass succeeds even for 1-shot batches.
            return images.sum() * 0.0
        perm = torch.randperm(images.shape[0], device=images.device)
        half = images.shape[0] // 2
        support = images[perm[:half]]
        query = images[perm[half : 2 * half]]
        return self.registration.compute_registration_loss(support, query)

