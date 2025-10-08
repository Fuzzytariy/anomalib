"""Differentiable feature registration modules with GraphCore support.

The registration network aligns support and query feature maps so that
subsequent distance computations operate on spatially consistent patch
representations.  The implementation includes GraphCore for rotation-invariant
feature extraction and provides:

* pyramid-style registration blocks with learned optical-flow predictions
* symmetric cosine-similarity losses that encourage cycle consistency
* hooks to keep track of running prototypes for each feature level
* GraphCore GCN for rotation-invariant feature extraction
"""

from __future__ import annotations

from collections.abc import Sequence
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
from torch.nn import functional as F

from anomalib.models.components.feature_extractors import TimmFeatureExtractor
from anomalib.models.components.graphcore import MultiLayerGraphCore


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

        # Ensure both tensors are on the same device before concatenation
        support, query = _ensure_same_device(support, query)
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
    graphcore_features: Dict[str, Tensor] | None = None


class FeatureRegistrationModule(nn.Module):
    """Extract and register multi-level CNN features with GraphCore support.

    The module stores running prototypes for each level of the backbone and can
    align arbitrary support/query batches against these prototypes. Includes
    GraphCore for rotation-invariant feature extraction.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        update_momentum: float = 0.1,
        use_graphcore: bool = True,
        graphcore_output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.extractor = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pre_trained)
        self.registration_blocks = nn.ModuleDict({
            layer: RegistrationBlock(channels=channels)
            for layer, channels in zip(layers, self.extractor.out_dims, strict=True)
        })
        self.layers = list(layers)
        self.update_momentum = update_momentum
        self.use_graphcore = use_graphcore
        self.graphcore_output_dim = graphcore_output_dim
        
        # Initialize GraphCore for layer2 and layer3 features
        if use_graphcore and len(layers) >= 2:
            layer2_dim = self.extractor.out_dims[0] if layers[0] == "layer2" else None
            layer3_dim = self.extractor.out_dims[1] if layers[1] == "layer3" else None
            if layer2_dim and layer3_dim:
                self.graphcore = MultiLayerGraphCore(
                    layer2_dim=layer2_dim,
                    layer3_dim=layer3_dim,
                    output_dim=graphcore_output_dim
                )
            else:
                self.graphcore = None
        else:
            self.graphcore = None
            
        self.register_buffer("prototypes_initialized", torch.tensor(False), persistent=True)
        self.prototypes: dict[str, Tensor] = {}


    def extract(self, images: Tensor) -> dict[str, Tensor]:
        """Extract backbone features."""

        return {layer: feat for layer, feat in self.extractor(images).items()}

    def initialise_prototypes(self, features: dict[str, Tensor]) -> None:
        """Initialise running prototypes from features."""


        
        # Ensure all layers are initialized
        for layer in self.layers:
            if layer in features:
                self.prototypes[layer] = features[layer].mean(dim=0, keepdim=True).detach()

            else:
                # Initialize with zeros if layer not in features
                if features:
                    first_feat = next(iter(features.values()))
                    feat_shape = first_feat.shape
                    self.prototypes[layer] = torch.zeros(feat_shape[1:], device=first_feat.device).unsqueeze(0)

                else:
                    # Fallback if no features available
                    self.prototypes[layer] = torch.zeros(1, 1, 1, 1, device=torch.device('cpu'))

        

        self.prototypes_initialized = torch.tensor(True, device=features[self.layers[0]].device if features else torch.device('cpu'))


    def _align_to_query(self, support: dict[str, Tensor], query: dict[str, Tensor]) -> RegistrationOutputs:
        aligned_support: dict[str, Tensor] = {}
        graphcore_features = None
        
        for layer in self.layers:
            reg_block = self.registration_blocks[layer]
            aligned_support[layer] = reg_block(support[layer], query[layer])
            
        # Apply GraphCore if available
        if self.graphcore is not None and "layer2" in support and "layer3" in support:
            graphcore_features = self.graphcore(support["layer2"], support["layer3"])
            
        return RegistrationOutputs(
            support=support, 
            query=query, 
            aligned_support=aligned_support,
            graphcore_features=graphcore_features
        )

    def register_support_batch(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """Align a batch of support features to running prototypes and update them."""


        
        # Force reinitialization if prototypes dictionary is empty but flag is True
        if not self.prototypes_initialized or len(self.prototypes) == 0:
            self.initialise_prototypes(features)
            # Return original features when prototypes are first initialized
            return features

        aligned: dict[str, Tensor] = {}
        for layer in self.layers:

            prototype = self.prototypes[layer]
            support = features[layer]
            prototype_expanded = prototype.expand_as(support)
            aligned[layer] = self.registration_blocks[layer](support, prototype_expanded)
            momentum = self.update_momentum
            proto_update = aligned[layer].mean(dim=0, keepdim=True).detach()
            self.prototypes[layer] = (1 - momentum) * prototype + momentum * proto_update
        return aligned

    def align_query_to_prototypes(self, query_features: dict[str, Tensor]) -> RegistrationOutputs:
        """Align stored prototypes to a new query batch using distribution statistics."""

        if not self.prototypes_initialized:
            self.initialise_prototypes(query_features)

        # 使用分布统计而不是原型扩展
        # 对于每个查询样本，我们使用相同的原型（基于记忆库的统计分布）
        support = {}
        for layer in self.layers:
            prototype = self.prototypes[layer]
            query_feat = query_features[layer]
            
            # 对于每个查询样本，使用相同的原型（基于记忆库的多元高斯分布）
            # 这样避免了批次大小不匹配的问题
            if prototype.size(0) != query_feat.size(0):
                # 使用第一个原型作为基础，扩展到查询批次大小
                base_prototype = prototype[0:1]  # 取第一个原型
                support[layer] = base_prototype.repeat(query_feat.size(0), 1, 1, 1)
            else:
                support[layer] = prototype
                
        return self._align_to_query(support=support, query=query_features)

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

