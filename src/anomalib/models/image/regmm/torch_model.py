"""Siamese 配准网络集成到 PatchCore 的 PyTorch 模型实现."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from anomalib.models.components import DynamicBufferMixin
from anomalib.models.image.regmm.siamese import SiameseRegistrationNetwork

logger = logging.getLogger(__name__)


class SiamesePatchcoreModel(DynamicBufferMixin, nn.Module):
    """PatchCore 特征提取器适配 RegAD Siamese 注册网络."""

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        siamese_weights: str | None = None,
        stn_enabled: bool = True,
        projection_dim: int = 128,
        distance_metric: str = "euclidean",
        stn_mode: str = "rotation_scale",
    ) -> None:
        super().__init__()
        self.tiler = None

        self.backbone = backbone
        self.layers = list(layers)
        self.num_neighbors = num_neighbors
        self.siamese_weights = siamese_weights
        self.stn_enabled = stn_enabled
        self.projection_dim = projection_dim
        self.distance_metric = distance_metric
        self.stn_mode = stn_mode

        if not self.stn_enabled:
            msg = "SiamesePatchcoreModel requires stn_enabled=True to use the registration backbone."
            raise ValueError(msg)

        siamese_backbone = self.backbone
        if siamese_backbone != "resnet18":
            logger.warning(
                "SiameseRegistrationNetwork only supports 'resnet18'. Received '%s' — falling back to 'resnet18'.",
                siamese_backbone,
            )
            siamese_backbone = "resnet18"

        self.feature_extractor = SiameseRegistrationNetwork(
            backbone=siamese_backbone,
            pre_trained=pre_trained,
            layers=self.layers,
            stn_enabled=self.stn_enabled,
            stn_mode=self.stn_mode,
        )

        if self.siamese_weights:
            self.load_siamese_weights(self.siamese_weights)

        self.feature_extractor.eval()

        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

        self.anomaly_map_generator = AnomalyMapGenerator()
        self.memory_bank: torch.Tensor
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store: list[torch.Tensor] = []
        self.register_buffer("memory_bank_norm", torch.empty(0))
        self.register_buffer("memory_bank_maha", torch.empty(0))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inv_cov", torch.empty(0))

    def load_siamese_weights(self, weights_path: str) -> None:
        """Load RegAD Siamese weights, ignoring incompatible tensors."""

        try:
            device = next(self.feature_extractor.parameters()).device
            ckpt = torch.load(weights_path, map_location=device)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            target_state = self.feature_extractor.state_dict()
            filtered_state: Dict[str, torch.Tensor] = OrderedDict()
            skipped: list[str] = []

            allowed_prefixes = ("feature_extractor.", "encoder.", "predictor.")

            for key, value in state.items():
                if key.startswith("model.siamese_net."):
                    new_key = key[len("model.siamese_net."):]
                elif key.startswith("siamese_net."):
                    new_key = key[len("siamese_net."):]
                elif key.startswith("model."):
                    new_key = key[len("model."):]
                else:
                    new_key = key

                if not new_key.startswith(allowed_prefixes):
                    continue

                if new_key not in target_state:
                    skipped.append(new_key)
                    continue

                if target_state[new_key].shape != value.shape:
                    skipped.append(new_key)
                    continue

                filtered_state[new_key] = value

            missing, unexpected = self.feature_extractor.load_state_dict(filtered_state, strict=False)

            logger.info("✓ 加载 Siamese 权重: %s", weights_path)
            logger.info(
                "   载入参数: %d | missing: %d | unexpected: %d | skipped: %d",
                len(filtered_state),
                len(missing),
                len(unexpected),
                len(skipped),
            )
            if skipped:
                logger.debug("跳过以下键（shape 不匹配或不存在）: %s", skipped)
        except Exception as exc:  # noqa: BLE001
            logger.warning("✗ 加载 Siamese 权重失败: %s", exc)
            logger.warning("✓ 将使用随机初始化的权重")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """前向传播."""

        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            feats_dict = self.feature_extractor.extract_features(input_tensor)

        layer_features: dict[str, torch.Tensor] = {}
        if len(feats_dict) == 0:
            raise ValueError("Siamese feature extractor returned no features. Check configuration.")
        for layer in self.layers:
            if layer in feats_dict:
                layer_features[layer] = feats_dict[layer]
            else:
                fallback = list(feats_dict.values())[-1]
                layer_features[layer] = fallback
        pooled = {name: self.feature_pooler(feat) for name, feat in layer_features.items()}
        embedding = None
        for name in self.layers:
            feat = pooled[name]
            if embedding is None:
                embedding = feat
            else:
                feat = F.interpolate(feat, size=embedding.shape[-2:], mode="bilinear", align_corners=False)
                embedding = torch.cat([embedding, feat], dim=1)

        if embedding is None:
            raise RuntimeError("Failed to build embedding from Siamese features.")

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        if self.memory_bank.size(0) == 0:
            raise ValueError("Memory bank is empty. Cannot provide anomaly scores")

        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        from anomalib.data import InferenceBatch

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """重塑嵌入张量（继承自 PatchCore）。"""

        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """最近邻搜索，根据配置的距离度量计算距离。"""

        if self.distance_metric == "euclidean":
            distances = self.euclidean_dist(embedding, self.memory_bank)
        elif self.distance_metric == "cosine":
            if self.memory_bank_norm.numel() == 0:
                raise ValueError("Cosine metric requires memory_bank_norm; call subsample_embedding() first.")
            emb_norm = F.normalize(embedding, p=2, dim=1)
            sim = torch.matmul(emb_norm, self.memory_bank_norm.t())
            distances = 1.0 - sim
        elif self.distance_metric in ("mahalanobis", "mahpp"):
            if self.inv_cov.numel() == 0 or self.mean.numel() == 0:
                raise ValueError("Mahalanobis metric requires mean and inv_cov; call subsample_embedding() first.")
            emb_in = F.normalize(embedding, p=2, dim=1) if self.distance_metric == "mahpp" else embedding
            q = (emb_in - self.mean) @ self.inv_cov
            if self.memory_bank_maha.numel() == 0:
                raise ValueError("Mahalanobis metric requires memory_bank_maha; call subsample_embedding() first.")
            distances = self.euclidean_dist(q, self.memory_bank_maha)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """欧几里得距离计算（继承自 PatchCore）。"""

        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """计算异常分数（继承自 PatchCore）。"""

        if self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        max_patches = torch.argmax(patch_scores, dim=1)
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        score = patch_scores[torch.arange(batch_size), max_patches]
        nn_index = locations[torch.arange(batch_size), max_patches]
        nn_sample = self.memory_bank[nn_index, :]
        memory_bank_effective_size = self.memory_bank.shape[0]
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        if self.distance_metric == "euclidean":
            distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        elif self.distance_metric == "cosine":
            q = F.normalize(max_patches_features, p=2, dim=1).unsqueeze(1)
            mem = self.memory_bank_norm[support_samples]
            distances = 1.0 - torch.matmul(q, mem.transpose(-2, -1))
        elif self.distance_metric in ("mahalanobis", "mahpp"):
            q = max_patches_features
            if self.distance_metric == "mahpp":
                q = F.normalize(q, p=2, dim=1)
            qw = (q - self.mean) @ self.inv_cov
            memw = self.memory_bank_maha[support_samples]
            distances = self.euclidean_dist(qw.unsqueeze(1), memw)
        else:
            raise ValueError(f"Unsupported metric {self.distance_metric}")
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        return weights * score

    def subsample_embedding(self, sampling_ratio: float) -> None:
        """子采样嵌入（继承自 PatchCore）。"""

        if len(self.embedding_store) == 0:
            raise ValueError("Embedding store is empty. Cannot perform coreset selection.")

        from anomalib.models.components import KCenterGreedy

        self.memory_bank = torch.vstack(self.embedding_store)
        self.embedding_store.clear()

        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

        if self.distance_metric == "cosine":
            self.memory_bank_norm = F.normalize(self.memory_bank, p=2, dim=1)
        elif self.distance_metric in ("mahalanobis", "mahpp"):
            memory = F.normalize(self.memory_bank, p=2, dim=1) if self.distance_metric == "mahpp" else self.memory_bank
            self.mean = memory.mean(dim=0)
            x_centered = memory - self.mean
            cov = x_centered.t().matmul(x_centered) / max(x_centered.size(0) - 1, 1)
            eps = 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            cov = cov + eps
            self.inv_cov = torch.linalg.pinv(cov)
            self.memory_bank_maha = (memory - self.mean) @ self.inv_cov
        else:
            device = self.memory_bank.device
            self.memory_bank_norm = torch.empty(0, device=device)
            self.memory_bank_maha = torch.empty(0, device=device)
            self.mean = torch.empty(0, device=device)
            self.inv_cov = torch.empty(0, device=device)
