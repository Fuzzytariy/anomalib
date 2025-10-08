"""RegAD Siamese registration backbone and losses.

This module ports the Siamese encoder/predictor branches, the spatial
transformer network (STN) equipped ResNet backbone, and the cosine/L2 losses
from the `MediaBrain-SJTU/RegAD` reference implementation.  The code is adapted
so that it can be consumed by the RegMM PatchCore integration while otherwise
staying faithful to the original project structure.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

__all__ = [
    "Encoder",
    "Predictor",
    "STNModule",
    "ResNetWithSTN",
    "stn_net",
    "SiameseRegistrationNetwork",
    "SiamesePretrainModel",
    "CosLoss",
    "L2Loss",
    "N_PARAMS",
]


# ---------------------------------------------------------------------------
# Utilities shared with the original RegAD implementation


N_PARAMS = {
    "affine": 6,
    "translation": 2,
    "rotation": 1,
    "scale": 2,
    "shear": 2,
    "rotation_scale": 3,
    "translation_scale": 4,
    "rotation_translation": 3,
    "rotation_translation_scale": 5,
}


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding."""

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ---------------------------------------------------------------------------
# Siamese encoder & predictor branches


class Encoder(nn.Module):
    """Channel preserving encoder copied from RegAD."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = conv1x1(in_planes=1024, out_planes=1024)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.conv2 = conv1x1(in_planes=1024, out_planes=1024)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU()

        self.conv3 = conv1x1(in_planes=1024, out_planes=1024)
        self.bn3 = nn.BatchNorm2d(1024)
        self.relu3 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        out = self.conv3(x)
        return out


class Predictor(nn.Module):
    """Predictor head (also channel preserving) from RegAD."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = conv1x1(in_planes=1024, out_planes=1024)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.conv2 = conv1x1(in_planes=1024, out_planes=1024)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        out = self.conv2(x)
        return out


# ---------------------------------------------------------------------------
# Spatial Transformer Network components


class STNModule(nn.Module):
    """Spatial transformer module reproduced from the RegAD codebase."""

    def __init__(self, in_channels: int, block_index: int, stn_mode: str) -> None:
        super().__init__()

        if stn_mode not in N_PARAMS:
            raise ValueError(f"Unsupported stn_mode '{stn_mode}'. Choices: {list(N_PARAMS)}")

        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]
        # In RegAD the feature-map size per block follows 56 // (4 * block_index).
        self.feat_size = 56 // (4 * block_index)

        self.conv = nn.Sequential(
            conv3x3(in_planes=in_channels, out_planes=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv3x3(in_planes=64, out_planes=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * self.feat_size * self.feat_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.stn_n_params),
        )

        # Parameter initialisation mirrors the RegAD reference.
        nn.init.zeros_(self.fc[2].weight)
        if self.stn_mode == "affine":
            bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        elif self.stn_mode in {"translation", "shear"}:
            bias = torch.tensor([0, 0], dtype=torch.float)
        elif self.stn_mode == "scale":
            bias = torch.tensor([1, 1], dtype=torch.float)
        elif self.stn_mode == "rotation":
            bias = torch.tensor([0], dtype=torch.float)
        elif self.stn_mode == "rotation_scale":
            bias = torch.tensor([0, 1, 1], dtype=torch.float)
        elif self.stn_mode == "translation_scale":
            bias = torch.tensor([0, 0, 1, 1], dtype=torch.float)
        elif self.stn_mode == "rotation_translation":
            bias = torch.tensor([0, 0, 0], dtype=torch.float)
        else:  # rotation_translation_scale
            bias = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float)
        self.fc[2].bias = nn.Parameter(bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        conv_x = self.conv(x)
        theta = self.fc(conv_x.view(batch_size, -1))
        theta_affine = self._build_theta(theta, batch_size, x.device)
        grid = F.affine_grid(theta_affine, x.shape, align_corners=False)
        transformed = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
        return transformed, theta_affine

    def _build_theta(self, theta: Tensor, batch_size: int, device: torch.device) -> Tensor:
        """Construct a 2x3 affine matrix according to the configured mode."""

        if self.stn_mode == "affine":
            return theta.view(batch_size, 2, 3)

        base = torch.zeros(batch_size, 2, 3, device=device, dtype=torch.float32)
        base[:, 0, 0] = 1.0
        base[:, 1, 1] = 1.0

        if self.stn_mode == "translation":
            base[:, 0, 2] = theta[:, 0]
            base[:, 1, 2] = theta[:, 1]
        elif self.stn_mode == "rotation":
            angle = theta[:, 0]
            base[:, 0, 0] = torch.cos(angle)
            base[:, 0, 1] = -torch.sin(angle)
            base[:, 1, 0] = torch.sin(angle)
            base[:, 1, 1] = torch.cos(angle)
        elif self.stn_mode == "scale":
            base[:, 0, 0] = theta[:, 0]
            base[:, 1, 1] = theta[:, 1]
        elif self.stn_mode == "shear":
            base[:, 0, 1] = theta[:, 0]
            base[:, 1, 0] = theta[:, 1]
        elif self.stn_mode == "rotation_scale":
            angle = theta[:, 0]
            base[:, 0, 0] = torch.cos(angle) * theta[:, 1]
            base[:, 0, 1] = -torch.sin(angle)
            base[:, 1, 0] = torch.sin(angle)
            base[:, 1, 1] = torch.cos(angle) * theta[:, 2]
        elif self.stn_mode == "translation_scale":
            base[:, 0, 2] = theta[:, 0]
            base[:, 1, 2] = theta[:, 1]
            base[:, 0, 0] = theta[:, 2]
            base[:, 1, 1] = theta[:, 3]
        elif self.stn_mode == "rotation_translation":
            angle = theta[:, 0]
            base[:, 0, 0] = torch.cos(angle)
            base[:, 0, 1] = -torch.sin(angle)
            base[:, 1, 0] = torch.sin(angle)
            base[:, 1, 1] = torch.cos(angle)
            base[:, 0, 2] = theta[:, 1]
            base[:, 1, 2] = theta[:, 2]
        else:  # rotation_translation_scale
            angle = theta[:, 0]
            base[:, 0, 0] = torch.cos(angle) * theta[:, 3]
            base[:, 0, 1] = -torch.sin(angle)
            base[:, 1, 0] = torch.sin(angle)
            base[:, 1, 1] = torch.cos(angle) * theta[:, 4]
            base[:, 0, 2] = theta[:, 1]
            base[:, 1, 2] = theta[:, 2]

        return base


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetWithSTN(nn.Module):
    """Wide-ResNet-50-2 backbone augmented with STN modules (RegAD)."""

    def __init__(
        self,
        stn_mode: str,
        block: type[nn.Module] = Bottleneck,
        layers: Iterable[int] = (3, 4, 6, 3),
        groups: int = 1,
        width_per_group: int = 128,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.stn1 = STNModule(64 * block.expansion, 1, stn_mode)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.stn2 = STNModule(128 * block.expansion, 2, stn_mode)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.stn3 = STNModule(256 * block.expansion, 3, stn_mode)

        # placeholders populated after forward pass
        self.stn1_output: Tensor | None = None
        self.stn2_output: Tensor | None = None
        self.stn3_output: Tensor | None = None

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _affine_inverse(theta: Tensor) -> Tensor:
        batch_size = theta.size(0)
        device = theta.device
        bottom_row = torch.tensor([0, 0, 1], dtype=theta.dtype, device=device)
        bottom = bottom_row.view(1, 1, 3).repeat(batch_size, 1, 1)
        full = torch.cat([theta, bottom], dim=1)
        inv = torch.inverse(full)
        return inv[:, :2, :]

    @staticmethod
    def _fixstn(x: Tensor, theta: Tensor) -> Tensor:
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x, theta1 = self.stn1(x)
        inv_theta1 = self._affine_inverse(theta1.detach())
        self.stn1_output = self._fixstn(x.detach(), inv_theta1)

        x = self.layer2(x)
        x, theta2 = self.stn2(x)
        inv_theta2 = self._affine_inverse(theta2.detach())
        self.stn2_output = self._fixstn(self._fixstn(x.detach(), inv_theta2), inv_theta1)

        x = self.layer3(x)
        out, theta3 = self.stn3(x)
        inv_theta3 = self._affine_inverse(theta3.detach())
        self.stn3_output = self._fixstn(self._fixstn(self._fixstn(out.detach(), inv_theta3), inv_theta2), inv_theta1)

        return out


def stn_net(stn_mode: str, pretrained: bool = True) -> ResNetWithSTN:
    """Construct the STN-equipped Wide-ResNet-50-2 backbone."""

    model = ResNetWithSTN(stn_mode, Bottleneck, [3, 4, 6, 3])
    if pretrained:
        weights = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).state_dict()
        model_dict = model.state_dict()
        filtered = {k: v for k, v in weights.items() if k in model_dict}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    return model


# ---------------------------------------------------------------------------
# Siamese wrapper compatible with RegMM PatchCore


@dataclass
class SiameseConfig:
    stn_mode: str = "rotation_scale"


class SiameseRegistrationNetwork(nn.Module):
    """Wraps the RegAD Siamese components for RegMM integration."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        layers: Optional[List[str]] = None,
        stn_enabled: bool = True,
        stn_mode: str = "rotation_scale",
    ) -> None:
        super().__init__()
        if backbone != "wide_resnet50_2":
            raise ValueError("RegAD SiameseRegistrationNetwork only supports wide_resnet50_2 backbone")

        self.layers = layers or ["layer1", "layer2", "layer3"]
        if not stn_enabled:
            raise ValueError("RegAD SiameseRegistrationNetwork requires the STN-enabled backbone")
        self.stn_enabled = stn_enabled
        self.config = SiameseConfig(stn_mode=stn_mode)

        # In the RegAD implementation the STN backbone, encoder and predictor are
        # independent modules.  We expose them here as submodules so that the
        # weight loading logic in ``torch_model.py`` can locate the parameters.
        if self.stn_enabled:
            self.feature_extractor = stn_net(self.config.stn_mode, pretrained=pre_trained)
        else:
            # fall back to the vanilla Wide-ResNet-50-2 without spatial alignment
            self.feature_extractor = wide_resnet50_2(
                weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pre_trained else None
            )
        self.encoder = Encoder()
        self.predictor = Predictor()

    def forward(self, query: Tensor, support: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if support is None:
            support = query

        query_feat = self._extract_backbone(query)
        support_feat = self._extract_backbone(support)

        z1 = self.encoder(query_feat)
        z2 = self.encoder(support_feat)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {
            "query_backbone": query_feat,
            "support_backbone": support_feat,
            "z1": z1,
            "z2": z2,
            "p1": p1,
            "p2": p2,
        }

    def _extract_backbone(self, x: Tensor) -> Tensor:
        if isinstance(self.feature_extractor, ResNetWithSTN):
            return self.feature_extractor(x)
        return self.feature_extractor(x)

    def extract_features(self, x: Tensor) -> Dict[str, Tensor]:
        if not isinstance(self.feature_extractor, ResNetWithSTN):
            raise RuntimeError("extract_features requires the STN-enabled backbone")

        _ = self.feature_extractor(x)
        outputs: Dict[str, Tensor] = OrderedDict()
        if "layer1" in self.layers and self.feature_extractor.stn1_output is not None:
            outputs["layer1"] = self.feature_extractor.stn1_output
        if "layer2" in self.layers and self.feature_extractor.stn2_output is not None:
            outputs["layer2"] = self.feature_extractor.stn2_output
        if "layer3" in self.layers and self.feature_extractor.stn3_output is not None:
            outputs["layer3"] = self.feature_extractor.stn3_output
        return outputs


class SiamesePretrainModel(nn.Module):
    """Simple wrapper computing the symmetric cosine loss used in RegAD."""

    def __init__(self, stn_mode: str = "rotation_scale", pre_trained: bool = True) -> None:
        super().__init__()
        self.siamese_net = SiameseRegistrationNetwork(
            backbone="wide_resnet50_2",
            pre_trained=pre_trained,
            stn_mode=stn_mode,
        )

    def forward(self, query: Tensor, support: Tensor) -> Dict[str, Tensor]:
        outputs = self.siamese_net(query, support)

        loss_forward = CosLoss(outputs["p1"], outputs["z2"], mean=True)
        loss_backward = CosLoss(outputs["p2"], outputs["z1"], mean=True)
        loss = 0.5 * (loss_forward + loss_backward)

        outputs["loss"] = loss
        outputs["loss_forward"] = loss_forward
        outputs["loss_backward"] = loss_backward
        return outputs

    def freeze_backbone(self) -> None:
        for param in self.siamese_net.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Loss functions


def L2Loss(data1: Tensor, data2: Tensor) -> Tensor:
    norm_data = torch.norm(data1 - data2, p=2, dim=1)
    return norm_data.mean()


def CosLoss(data1: Tensor, data2: Tensor, mean: bool = True) -> Tensor:
    data2 = data2.detach()
    cos = nn.CosineSimilarity(dim=1)
    if mean:
        return -cos(data1, data2).mean()
    return -cos(data1, data2)

