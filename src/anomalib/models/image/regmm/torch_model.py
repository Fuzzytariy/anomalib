"""Siamese配准网络集成到PatchCore的PyTorch模型实现

将SiameseRegistrationNetwork作为PatchCore的backbone，实现配准增强的异常检测。
"""

import torch
from torch import nn
from torch.nn import functional as F
from anomalib.models.components import DynamicBufferMixin, TimmFeatureExtractor
from anomalib.models.image.regmm.siamese import SiameseRegistrationNetwork
class SiamesePatchcoreModel(DynamicBufferMixin, nn.Module):



    def __init__(
        self,
        layers: list[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        siamese_weights: str = None,
        stn_enabled: bool = True,
        projection_dim: int = 128,
        distance_metric: str = "euclidean",
    ) -> None:
        super().__init__()
        self.tiler = None

        self.backbone = backbone
        self.layers = layers
        self.num_neighbors = num_neighbors
        self.siamese_weights = siamese_weights
        self.stn_enabled = stn_enabled
        self.projection_dim = projection_dim
        # Distance metric: one of 'euclidean', 'cosine', 'mahalanobis', 'mahpp'
        # - euclidean: L2 distance
        # - cosine: 1 - cosine similarity
        # - mahalanobis: Mahalanobis distance computed on raw features
        # - mahpp: Mahalanobis++ distance on ℓ2‑normalised features
        self.distance_metric = distance_metric

        # 使用SiameseRegistrationNetwork作为特征提取器
        self.feature_extractor = SiameseRegistrationNetwork(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=self.layers,
            stn_enabled=self.stn_enabled,
            projection_dim=self.projection_dim
        )

        # 加载预训练权重
        if self.siamese_weights:
            self.load_siamese_weights(self.siamese_weights)


        # 保持Siamese网络在eval模式
        self.feature_extractor.eval()

        # 继承PatchCore的其他组件
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.memory_bank: torch.Tensor
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store: list[torch.Tensor] = []
        # Additional buffers for alternative distance metrics
        self.register_buffer("memory_bank_norm", torch.empty(0))
        self.register_buffer("memory_bank_maha", torch.empty(0))
        # Mean and inverse covariance for Mahalanobis distances
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inv_cov", torch.empty(0))

    # torch_model.py
    def load_siamese_weights(self, weights_path: str) -> None:
        try:
            device = next(self.feature_extractor.parameters()).device
            ckpt = torch.load(weights_path, map_location=device)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            new_state = {}
            for k, v in state.items():
                # 统一去前缀
                if k.startswith("model.siamese_net."):
                    nk = k[len("model.siamese_net."):]
                elif k.startswith("siamese_net."):
                    nk = k[len("siamese_net."):]
                elif k.startswith("model."):
                    nk = k[len("model."):]
                else:
                    nk = k
                # 只接收属于 SiameseRegistrationNetwork 的模块
                if nk.startswith(("feature_extractor.", "stn.", "projection_head.", "predictor.", "encoder.")):
                    new_state[nk] = v

            missing, unexpected = self.feature_extractor.load_state_dict(new_state, strict=False)
            print(f"✓ 加载Siamese权重: {weights_path}")
            print(f"   加载参数数: {len(new_state)} | missing: {len(missing)} | unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"✗ 加载Siamese权重失败: {e}")
            print("✓ 将使用随机初始化的权重")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            input_tensor (torch.Tensor): 输入张量，形状为(batch_size, 3, height, width)

        Returns:
            torch.Tensor: 特征嵌入或异常检测结果
        """
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            # 使用配准网络提取多层特征。extract_features 返回的是对齐后的特征
            # 字典键是层名，值是形状 (B, C, H, W)
            feats_dict = self.feature_extractor.extract_features(input_tensor)

        # 填充缺失层：如果某些层在结果中不存在，使用最后一个可用层替代。
        layer_features: dict[str, torch.Tensor] = {}
        if len(feats_dict) == 0:
            raise ValueError("Siamese feature extractor returned no features. Check configuration.")
        for idx, layer in enumerate(self.layers):
            if layer in feats_dict:
                layer_features[layer] = feats_dict[layer]
            else:
                # 如果指定的层不存在，使用 deepest available layer
                # 按字典顺序选择最后的特征作为替代
                fallback = list(feats_dict.values())[-1]
                layer_features[layer] = fallback
        pooled = {l: self.feature_pooler(f) for l, f in layer_features.items()}
        emb = None
        for l in self.layers:
            feat = pooled[l]
            if emb is None:
                emb = feat
            else:
                feat = F.interpolate(feat, size=emb.shape[-2:], mode="bilinear", align_corners=False)
                emb = torch.cat([emb, feat], dim=1)
        embedding = emb

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        # 异常检测逻辑（继承自PatchCore）
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
        """重塑嵌入张量（继承自PatchCore）"""
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """最近邻搜索，根据配置的距离度量计算距离。

        Args:
            embedding (torch.Tensor): Query embeddings of shape ``(N, D)``.
            n_neighbors (int): Number of nearest neighbors to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (distances, indices) where
            ``distances`` has shape ``(N, k)`` and contains the distances to
            the nearest neighbors, and ``indices`` contains the indices in
            the memory bank.
        """
        # Compute pairwise distances according to the selected metric
        if self.distance_metric == "euclidean":
            distances = self.euclidean_dist(embedding, self.memory_bank)
        elif self.distance_metric == "cosine":
            # Normalise query embeddings and memory bank
            if self.memory_bank_norm.numel() == 0:
                raise ValueError("Cosine metric requires memory_bank_norm; call subsample_embedding() first.")
            emb_norm = F.normalize(embedding, p=2, dim=1)
            # Compute cosine similarity and convert to distance
            sim = torch.matmul(emb_norm, self.memory_bank_norm.t())
            distances = 1.0 - sim
        elif self.distance_metric in ("mahalanobis", "mahpp"):
            # Whiten query features: subtract mean and multiply by inv_cov
            if self.inv_cov.numel() == 0 or self.mean.numel() == 0:
                raise ValueError("Mahalanobis metric requires mean and inv_cov; call subsample_embedding() first.")
            if self.distance_metric == "mahpp":
                emb_in = F.normalize(embedding, p=2, dim=1)
            else:
                emb_in = embedding
            q = (emb_in - self.mean) @ self.inv_cov
            # Use whitened memory bank for distance computation
            if self.memory_bank_maha.numel() == 0:
                raise ValueError("Mahalanobis metric requires memory_bank_maha; call subsample_embedding() first.")
            distances = self.euclidean_dist(q, self.memory_bank_maha)


        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        # Retrieve nearest neighbors
        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """欧几里得距离计算（继承自PatchCore）"""
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
        """计算异常分数（继承自PatchCore）"""
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
        # torch_model.py - SiamesePatchcoreModel.compute_anomaly_score
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
        """子采样嵌入（继承自PatchCore）"""
        if len(self.embedding_store) == 0:
            raise ValueError("Embedding store is empty. Cannot perform coreset selection.")

        from anomalib.models.components import KCenterGreedy
        # Stack all collected embeddings into memory_bank: shape (N, D)
        self.memory_bank = torch.vstack(self.embedding_store)
        # Clear store after stacking
        self.embedding_store.clear()

        # Perform coreset sampling to reduce memory usage
        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

        # Precompute additional statistics for non‑euclidean metrics
        # Cosine: normalise memory bank vectors
        if self.distance_metric == "cosine":
            self.memory_bank_norm = F.normalize(self.memory_bank, p=2, dim=1)
        # Mahalanobis: compute mean and inverse covariance, and whiten memory bank
        elif self.distance_metric in ("mahalanobis", "mahpp"):
            # If Mahalanobis++: normalise first
            if self.distance_metric == "mahpp":
                memory = F.normalize(self.memory_bank, p=2, dim=1)
            else:
                memory = self.memory_bank
            # Compute mean
            self.mean = memory.mean(dim=0)
            # Zero‑centered data
            X = memory - self.mean
            # Compute covariance matrix with small epsilon for stability
            # cov = X^T X / (n - 1)
            cov = X.t().matmul(X) / max(X.size(0) - 1, 1)
            eps = 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            cov = cov + eps
            # Pseudo‑inverse in case covariance is singular
            self.inv_cov = torch.linalg.pinv(cov)
            # Compute whitened memory bank: (x - mean) @ inv_cov
            self.memory_bank_maha = (memory - self.mean) @ self.inv_cov
        # Euclidean: no extra buffers needed, ensure other buffers are empty
        else:
            # For Euclidean, we clear auxiliary buffers to free memory
            self.memory_bank_norm = torch.empty(0, device=self.memory_bank.device)
            self.memory_bank_maha = torch.empty(0, device=self.memory_bank.device)
            self.mean = torch.empty(0, device=self.memory_bank.device)
            self.inv_cov = torch.empty(0, device=self.memory_bank.device)