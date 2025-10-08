"""Siamese配准网络实现

实现双支路的Siamese网络，包含可学习的空间变换网络(STN)用于特征对齐。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional

from anomalib.models.components.feature_extractors.timm import TimmFeatureExtractor


class SpatialTransformNetwork(nn.Module):
    """空间变换网络(STN)用于特征对齐
    
    在layer3特征后接入STN，学习空间变换参数来对齐特征图。
    """
    
    def __init__(self, input_channels: int):
        super().__init__()
        
        # 定位网络 - 预测空间变换参数
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 回归网络 - 预测仿射变换参数(6个参数)
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )
        
        # 初始化变换参数为恒等变换
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x: Tensor) -> Tensor:
        """应用空间变换到输入特征图"""
        # 预测变换参数
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # 创建网格并应用变换（改进参数设置）
        B, C, H, W = x.shape
        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
        x_transformed = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        
        return x_transformed


class SiameseRegistrationNetwork(nn.Module):
    """Siamese配准网络（SimSiam风格）
    
    双支路共享Backbone，在layer3后接入STN进行特征对齐。
    使用SimSiam结构防止特征塌陷。
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pre_trained: bool = True,
        layers: List[str] = ["layer1", "layer2", "layer3"],
        projection_dim: int = 128,
        stn_enabled: bool = True,
        predictor_dim: int = 512  # 预测器隐藏层维度
    ):
        super().__init__()
        
        self.backbone = backbone
        self.layers = layers
        self.stn_enabled = stn_enabled
        
        # 共享的特征提取器
        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers
        )
        
        # 获取layer3的输出通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.feature_extractor(dummy_input)
            layer3_channels = features["layer3"].shape[1]
        
        # 空间变换网络
        if stn_enabled:
            self.stn = SpatialTransformNetwork(layer3_channels)
        else:
            self.stn = None
        
        # 投影头 - 将特征映射到低维空间（使用LayerNorm避免批次大小问题）
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(layer3_channels, 512),
            nn.LayerNorm(512),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
            nn.LayerNorm(projection_dim)  # SimSiam风格：使用LayerNorm
        )
        
        # 预测器头 - 防止特征塌陷的关键组件
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Linear(predictor_dim, projection_dim)
        )
    
    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, stop_gradient: bool = True) -> Dict[str, Tensor]:
        """前向传播（SimSiam风格）
        
        Args:
            x1: 第一支路输入张量
            x2: 第二支路输入张量，如果为None则使用x1
            stop_gradient: 是否对第二支路使用stop-gradient
            
        Returns:
            包含两支路输出的字典
        """
        if x2 is None:
            x2 = x1
        
        # 提取特征
        features1 = self.feature_extractor(x1)
        features2 = self.feature_extractor(x2)
        
        # 获取layer3特征
        layer3_1 = features1["layer3"]
        layer3_2 = features2["layer3"]
        
        # 应用空间变换（如果启用）
        if self.stn_enabled and self.stn is not None:
            layer3_1 = self.stn(layer3_1)
            layer3_2 = self.stn(layer3_2)
        
        # 通过投影头
        z1 = self.projection_head(layer3_1)
        z2 = self.projection_head(layer3_2)
        
        # 归一化（SimSiam风格）
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # 应用预测器（只在第一支路）
        p1 = self.predictor(z1)
        
        # SimSiam关键：对第二支路使用stop-gradient
        if stop_gradient:
            z2 = z2.detach()  # 阻断梯度传播
        
        return {
            "p1": p1,  # 预测结果
            "z1": z1,  # 第一支路投影
            "z2": z2,  # 第二支路投影（可能detach）
            "features1": layer3_1,
            "features2": layer3_2
        }

    # ------------------------------------------------------------------
    # Feature extraction for anomaly detection
    #
    # During inference with PatchCore we do not require the SimSiam projection
    # vectors.  Instead we need intermediate CNN features from multiple
    # layers.  This helper extracts such features and applies the spatial
    # transformer on the deep layer when enabled.  It returns a
    # dictionary mapping layer names to aligned feature maps.  Only the
    # layers specified in ``self.layers`` are returned.  For ``layer3`` the
    # STN alignment is applied; other layers are returned unchanged.
    def extract_features(self, x: Tensor) -> Dict[str, Tensor]:
        """Extract backbone features for PatchCore.

        Args:
            x (Tensor): Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Dict[str, Tensor]: A mapping from layer names (e.g. ``"layer2"``,
            ``"layer3"``) to spatial feature maps after alignment.  Only
            layers listed in ``self.layers`` are included.
        """
        # Use the timm feature extractor to obtain all intermediate features.
        feats = self.feature_extractor(x)
        feat_dict: Dict[str, Tensor] = {}
        for layer in self.layers:
            if layer not in feats:
                # If the requested layer is missing, skip it.  This can
                # happen if ``self.layers`` includes keys not produced by
                # ``TimmFeatureExtractor``.  PatchCore will handle missing
                # layers by reusing the deepest available layer.
                continue
            feat = feats[layer]
            # Only apply STN alignment on the deepest layer (layer3) when
            # enabled.  Alignment of shallower layers did not show
            # consistent gains in preliminary experiments and introduces
            # additional overhead.
            if self.stn_enabled and self.stn is not None and layer == "layer3":
                feat = self.stn(feat)
            feat_dict[layer] = feat
        return feat_dict


class SiamesePretrainModel(nn.Module):
    """Siamese预训练模型（SimSiam风格）
    
    包含完整的训练流程和损失计算，使用SimSiam损失防止特征塌陷。
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pre_trained: bool = True,
        layers: List[str] = ["layer1", "layer2", "layer3"],
        projection_dim: int = 128,
        stn_enabled: bool = True,
        predictor_dim: int = 512
    ):
        super().__init__()
        
        self.siamese_net = SiameseRegistrationNetwork(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            projection_dim=projection_dim,
            stn_enabled=stn_enabled,
            predictor_dim=predictor_dim
        )
    
    def forward(self, x1: Tensor, x2: Tensor) -> Dict[str, Tensor]:
        """前向传播，计算SimSiam损失"""
        # 通过Siamese网络（使用stop-gradient）
        outputs = self.siamese_net(x1, x2, stop_gradient=True)
        
        # SimSiam损失：负余弦相似度
        p1 = outputs["p1"]  # 预测结果
        z2 = outputs["z2"]  # 目标（detached）
        
        # 计算负余弦相似度损失
        p1 = F.normalize(p1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # SimSiam损失：- (p1 · z2) 的均值
        loss = -(p1 * z2).sum(dim=1).mean()
        
        # 计算对称损失（交换x1和x2）
        outputs_sym = self.siamese_net(x2, x1, stop_gradient=True)
        p2 = F.normalize(outputs_sym["p1"], p=2, dim=1)
        z1 = F.normalize(outputs_sym["z2"], p=2, dim=1)
        
        loss_sym = -(p2 * z1).sum(dim=1).mean()
        
        # 总损失
        total_loss = 0.5 * (loss + loss_sym)
        
        # 计算余弦相似度用于监控（监控p1和z2的相似度，反映SimSiam训练效果）
        p1_z2_sim = F.cosine_similarity(p1, z2, dim=1)
        p2_z1_sim = F.cosine_similarity(p2, z1, dim=1)
        avg_simsiam_sim = 0.5 * (p1_z2_sim.mean() + p2_z1_sim.mean())
        
        outputs["loss"] = total_loss
        outputs["cosine_similarity"] = avg_simsiam_sim  # 使用SimSiam相似度而不是z1-z2相似度
        outputs["simsiam_loss"] = loss
        outputs["sym_loss"] = loss_sym
        outputs["p1_z2_sim"] = p1_z2_sim.mean()  # 单独监控p1-z2相似度
        outputs["p2_z1_sim"] = p2_z1_sim.mean()  # 单独监控p2-z1相似度
        
        return outputs
    
    def freeze_backbone(self):
        """冻结Backbone权重，但保持STN和投影头可训练"""
        # 只冻结特征提取器（backbone）
        for param in self.siamese_net.feature_extractor.parameters():
            param.requires_grad = False
        
        # STN和投影头保持可训练状态
        if self.siamese_net.stn is not None:
            for param in self.siamese_net.stn.parameters():
                param.requires_grad = True
        
        for param in self.siamese_net.projection_head.parameters():
            param.requires_grad = True
    
    def freeze_all(self):
        """冻结所有权重（Backbone、STN和投影头）"""
        # 冻结特征提取器
        for param in self.siamese_net.feature_extractor.parameters():
            param.requires_grad = False
        
        # 冻结STN（如果存在）
        if self.siamese_net.stn is not None:
            for param in self.siamese_net.stn.parameters():
                param.requires_grad = False
        
        # 冻结投影头
        for param in self.siamese_net.projection_head.parameters():
            param.requires_grad = False
    
    def get_trainable_parameters(self):
        """获取可训练参数（用于调试）"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        return trainable_params