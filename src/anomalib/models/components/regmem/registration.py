"""Differentiable feature registration modules.

The registration network aligns support and query feature maps so that
subsequent distance computations operate on spatially consistent patch
representations.  The implementation is intentionally lightweight to keep the
dependency surface small while still providing:

* pyramid-style registration blocks with learned optical-flow predictions
* symmetric cosine-similarity losses that encourage cycle consistency
* hooks to keep track of running prototypes for each feature level
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anomalib.models.components.feature_extractors import TimmFeatureExtractor


def _identity_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a normalised ``[-1, 1]`` sampling grid."""

    ys, xs = torch.linspace(-1, 1, steps=height, device=device, dtype=dtype), torch.linspace(
        -1, 1, steps=width, device=device, dtype=dtype
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


class RegistrationBlock(nn.Module):
    """Predict a dense flow field that warps support features onto query features."""

    def __init__(self, channels: int, hidden_channels: int = 128, flow_scale: float = 0.5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 2, kernel_size=3, padding=1, bias=True),
        )
        self.flow_scale = flow_scale

    def forward(self, support: Tensor, query: Tensor) -> Tensor:
        """Warp ``support`` to match ``query``.

        Args:
            support: Support feature map of shape ``(B, C, H, W)``.
            query: Query feature map of shape ``(B, C, H, W)``.
        Returns:
            Tensor: Support feature map spatially aligned to the query.
        """

        if support.shape != query.shape:
            msg = "Support and query feature maps must share the same shape."
            raise ValueError(msg)

        features = torch.cat([support, query], dim=1)
        flow = torch.tanh(self.encoder(features)) * self.flow_scale
        grid = _identity_grid(query.shape[-2], query.shape[-1], device=query.device, dtype=query.dtype)
        grid = grid.unsqueeze(0).expand(query.shape[0], -1, -1, -1)
        grid = grid + flow.permute(0, 2, 3, 1)

        aligned = F.grid_sample(support, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return aligned


@dataclass
class RegistrationOutputs:
    """Container for registration outputs per feature level."""

    support: Dict[str, Tensor]
    query: Dict[str, Tensor]
    aligned_support: Dict[str, Tensor]


class FeatureRegistrationModule(nn.Module):
    """Extract and register multi-level CNN features.

    The module stores running prototypes for each level of the backbone and can
    align arbitrary support/query batches against these prototypes.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        pre_trained: bool = True,
        update_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.extractor = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pre_trained)
        self.registration_blocks = nn.ModuleDict({
            layer: RegistrationBlock(channels=channels)
            for layer, channels in zip(layers, self.extractor.out_dims, strict=True)
        })
        self.layers = list(layers)
        self.update_momentum = update_momentum
        self.register_buffer("prototypes_initialized", torch.tensor(False), persistent=True)
        self.prototypes: dict[str, Tensor] = {}

    def extract(self, images: Tensor) -> dict[str, Tensor]:
        """Extract backbone features."""

        return {layer: feat for layer, feat in self.extractor(images).items()}

    def initialise_prototypes(self, features: dict[str, Tensor]) -> None:
        """Initialise running prototypes from features."""

        for layer, feat in features.items():
            self.prototypes[layer] = feat.mean(dim=0, keepdim=True).detach()
        self.prototypes_initialized = torch.tensor(True, device=features[self.layers[0]].device)

    def _align_to_query(self, support: dict[str, Tensor], query: dict[str, Tensor]) -> RegistrationOutputs:
        aligned_support: dict[str, Tensor] = {}
        for layer in self.layers:
            reg_block = self.registration_blocks[layer]
            aligned_support[layer] = reg_block(support[layer], query[layer])
        return RegistrationOutputs(support=support, query=query, aligned_support=aligned_support)

    def register_support_batch(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """Align a batch of support features to running prototypes and update them."""

        if not self.prototypes_initialized:
            self.initialise_prototypes(features)
            return {layer: feat.detach() for layer, feat in features.items()}

        aligned: dict[str, Tensor] = {}
        for layer in self.layers:
            prototype = self.prototypes[layer]
            support = features[layer]
            prototype_expanded = prototype.expand_as(support)
            aligned[layer] = self.registration_blocks[layer](support, prototype_expanded).detach()
            momentum = self.update_momentum
            proto_update = aligned[layer].mean(dim=0, keepdim=True)
            self.prototypes[layer] = (1 - momentum) * prototype + momentum * proto_update
        return aligned

    def align_query_to_prototypes(self, query_features: dict[str, Tensor]) -> RegistrationOutputs:
        """Align stored prototypes to a new query batch."""

        if not self.prototypes_initialized:
            self.initialise_prototypes(query_features)

        support = {layer: self.prototypes[layer].expand_as(query_features[layer]) for layer in self.layers}
        return self._align_to_query(support=support, query=query_features)

    def align_features_to_prototypes(self, query_features: dict[str, Tensor]) -> dict[str, Tensor]:
        """Warp query features into the prototype reference frame."""

        if not self.prototypes_initialized:
            self.initialise_prototypes(query_features)

        prototypes = {layer: self.prototypes[layer].expand_as(query_features[layer]) for layer in self.layers}
        outputs = self._align_to_query(support=query_features, query=prototypes)
        return outputs.aligned_support

    def compute_registration_loss(self, support_images: Tensor, query_images: Tensor) -> Tensor:
        """Symmetric cosine similarity loss for registration."""

        support_feats = self.extract(support_images)
        query_feats = self.extract(query_images)

        outputs = self._align_to_query(support_feats, query_feats)
        loss = 0.0
        for layer in self.layers:
            aligned = outputs.aligned_support[layer]
            query = outputs.query[layer]
            cos = F.cosine_similarity(aligned, query, dim=1)
            loss = loss + (1 - cos.mean())

        # Symmetric term (query->support)
        reverse_outputs = self._align_to_query(query_feats, support_feats)
        for layer in self.layers:
            aligned = reverse_outputs.aligned_support[layer]
            support = reverse_outputs.query[layer]
            cos = F.cosine_similarity(aligned, support, dim=1)
            loss = loss + (1 - cos.mean())

        return loss / (2 * len(self.layers))

