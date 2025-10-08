# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Modified for Siamese-Registration backbone + pluggable distance

import logging
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

# 注意：这里导入你集成好的 Siamese 版 PatchCore torch 模型
# 路径按你的工程放置来改（下面给到推荐路径）
from .torch_model import SiamesePatchcoreModel

logger = logging.getLogger(__name__)


class SiamesePatchcore(MemoryBankMixin, AnomalibModule):
    """PatchCore with Siamese-Registration backbone & pluggable distance."""

    def __init__(
        self,
        backbone: str = "resnet18",                 # 用在 SiameseRegistrationNetwork 里的 backbone
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        # === 新增：Siamese 配置 ===
        siamese_weights: str | None = None,    # 预训练对齐权重
        stn_enabled: bool = True,                   # 是否启用 STN
        projection_dim: int = 128,                  # 投影维度（若你的 Siamese 模块需要）
        # === 新增：可选距离度量 ===
        distance: str = "euclidean",                # ["euclidean","cosine","mahalanobis","mahalanobis_pp","mmd"]
        # 框架通用部件：
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model: SiamesePatchcoreModel = SiamesePatchcoreModel(
            layers=list(layers),
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
            siamese_weights=siamese_weights,
            stn_enabled=stn_enabled,
            projection_dim=projection_dim,
            distance_metric=distance,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio

    # 保持与原 PatchCore 一致
    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        center_crop_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        image_size = image_size or (256, 256)
        if center_crop_size is not None:
            if center_crop_size[0] > image_size[0] or center_crop_size[1] > image_size[1]:
                raise ValueError(f"Center crop size {center_crop_size} cannot be larger than image size {image_size}.")
            transform = Compose([
                Resize(image_size, antialias=True),
                CenterCrop(center_crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return PreProcessor(transform=transform)

    @staticmethod
    def configure_optimizers() -> None:
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        _ = self.model(batch.image)  # 只提特征并缓存到 embedding_store（与原 PatchCore 一致）
        # 返回一个 dummy loss 以兼容 Lightning（不做反传）
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(self.coreset_sampling_ratio)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        return PostProcessor()
