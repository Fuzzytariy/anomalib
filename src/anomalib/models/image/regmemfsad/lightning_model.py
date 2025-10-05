"""Lightning module that orchestrates the RegMem few-shot anomaly detector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.visualization import Visualizer

from .torch_model import RegMemFewShotModel


class RegMemFSAD(MemoryBankMixin, AnomalibModule):
    """Few-shot anomaly detection model with feature registration and memory bank."""

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.05,
        num_neighbors: int = 5,
        distribution_weights: Sequence[float] = (0.5, 0.3, 0.2),
        mmd_sigma: float = 1.0,
        learning_rate: float = 1e-4,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        *,
        max_epochs: int = 50,
        gradient_clip_val: float = 0.0,
        num_sanity_val_steps: int = 0,
        trainer_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = RegMemFewShotModel(
            backbone=backbone,
            layers=tuple(layers),
            pre_trained=pre_trained,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            distribution_weights=tuple(distribution_weights),
            mmd_sigma=mmd_sigma,
        )
        self.learning_rate = learning_rate
        self._trainer_max_epochs = max_epochs
        self._trainer_gradient_clip_val = gradient_clip_val
        self._trainer_num_sanity_val_steps = num_sanity_val_steps
        self._trainer_overrides = dict(trainer_overrides or {})

    def on_train_start(self) -> None:
        self.model.reset()

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        loss = self.model.registration_loss(batch.image)
        self.log("train/registration_loss", loss, prog_bar=True)
        self.model.collect_support_features(batch.image)
        return loss

    def fit(self) -> None:
        self.model.build_memory_bank()

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs
        predictions = self.model.predict(batch.image)
        return batch.update(
            pred_score=predictions.pred_score,
            anomaly_map=predictions.anomaly_map,
            anomaly_maps=predictions.per_layer_maps,
        )

    def test_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.validation_step(batch, *args, **kwargs)

    def predict_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.validation_step(batch, *args, **kwargs)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        trainer_args: dict[str, Any] = {
            "gradient_clip_val": self._trainer_gradient_clip_val,
            "max_epochs": self._trainer_max_epochs,
            "num_sanity_val_steps": self._trainer_num_sanity_val_steps,
        }
        trainer_args.update(self._trainer_overrides)
        return trainer_args

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        return PostProcessor()

