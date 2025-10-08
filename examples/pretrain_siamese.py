"""Siamese配准网络预训练脚本

使用指定的配置文件和数据集路径进行Siamese网络预训练。
"""

import os
import yaml
import torch
torch.use_deterministic_algorithms(False)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from anomalib.data.pretrain_dataset import PretrainDataset, create_pretrain_dataloader
from anomalib.models.image.regmm.siamese import SiamesePretrainModel


class SiamesePretrainLightningModule(pl.LightningModule):
    """Siamese预训练的Lightning模块"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = SiamesePretrainModel(**config["model"]["init_args"])
        self.save_hyperparameters()
    
    def forward(self, x1, x2):
        return self.model(x1, x2)
    
    def training_step(self, batch, batch_idx):
        x1, x2, _ = batch  # 现在batch包含三个元素：(x1, x2, category)
        outputs = self(x1, x2)
        
        self.log("train_loss", outputs["loss"], prog_bar=True)
        self.log("train_cosine_sim", outputs["cosine_similarity"], prog_bar=True)
        self.log("train_simsiam_loss", outputs["simsiam_loss"], prog_bar=True)
        self.log("train_sym_loss", outputs["sym_loss"], prog_bar=True)
        self.log("train_p1_z2_sim", outputs["p1_z2_sim"], prog_bar=True)
        
        return outputs["loss"]
    
    def validation_step(self, batch, batch_idx):
        x1, x2, _ = batch  # 现在batch包含三个元素：(x1, x2, category)
        outputs = self(x1, x2)
        
        self.log("val_loss", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("val_cosine_sim", outputs["cosine_similarity"], prog_bar=True)
        self.log("val_simsiam_loss", outputs["simsiam_loss"], prog_bar=True)
        self.log("val_sym_loss", outputs["sym_loss"], prog_bar=True)
        self.log("val_p1_z2_sim", outputs["p1_z2_sim"], prog_bar=True)
        
        return outputs["loss"]
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["optimizer"]["init_args"]["lr"],
            weight_decay=self.config["optimizer"]["init_args"]["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config["trainer"]["max_epochs"]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """创建训练和验证数据加载器（支持SimSiam采样）"""
    data_config = config["data"]["init_args"]
    
    # 创建训练数据集
    train_dataset = PretrainDataset(
        root=data_config["root"],
        category=data_config["category"],
        split="train",
        image_size=tuple(data_config["image_size"])
    )
    
    # 创建验证数据集
    val_dataset = PretrainDataset(
        root=data_config["root"],
        category=data_config["category"],
        split="train",  # 使用训练集进行验证
        image_size=tuple(data_config["image_size"])
    )
    
    # 创建数据加载器（现在__getitem__直接返回不同图片对，无需自定义collate_fn）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        shuffle=config["train_dataloader"]["shuffle"],
        num_workers=config["train_dataloader"]["num_workers"],
        pin_memory=config["train_dataloader"]["pin_memory"],
        persistent_workers=True if config["train_dataloader"]["num_workers"] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_dataloader"]["batch_size"],
        shuffle=config["val_dataloader"]["shuffle"],
        num_workers=config["val_dataloader"]["num_workers"],
        pin_memory=config["val_dataloader"]["pin_memory"],
        persistent_workers=True if config["val_dataloader"]["num_workers"] > 0 else False
    )
    
    return train_loader, val_loader


def setup_callbacks(config: dict):
    """设置训练回调函数"""
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["callbacks"][0]["init_args"]["dirpath"],
        filename=config["callbacks"][0]["init_args"]["filename"],
        monitor=config["callbacks"][0]["init_args"]["monitor"],
        mode=config["callbacks"][0]["init_args"]["mode"],
        save_top_k=config["callbacks"][0]["init_args"]["save_top_k"],
        save_last=config["callbacks"][0]["init_args"]["save_last"]
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    early_stopping = EarlyStopping(
        monitor=config["callbacks"][1]["init_args"]["monitor"],
        patience=config["callbacks"][1]["init_args"]["patience"],
        mode=config["callbacks"][1]["init_args"]["mode"]
    )
    callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(
        logging_interval=config["callbacks"][2]["init_args"]["logging_interval"]
    )
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    """主训练函数"""
    # 加载配置文件
    config_path = "configs/pretrain/siamese_registration.yaml"
    config = load_config(config_path)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建模型
    model = SiamesePretrainLightningModule(config)
    
    # 设置回调函数
    callbacks = setup_callbacks(config)
    
    # 设置日志记录器
    logger = TensorBoardLogger(
        save_dir=config["logging"]["save_dir"],
        name=config["logging"]["name"],
        version=config["logging"]["version"]
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=config["trainer"]["enable_checkpointing"],
        enable_progress_bar=config["trainer"]["enable_progress_bar"],
        deterministic=False,  # 关闭确定性算法以避免grid_sample错误
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        val_check_interval=config["training"]["val_check_interval"],
        check_val_every_n_epoch=config["training"]["check_val_every_n_epoch"]
    )
    
    # 开始训练
    print("开始Siamese配准网络预训练...")
    print(f"使用类别: {config['data']['init_args']['category']}")
    print(f"数据集路径: {config['data']['init_args']['root']}")
    print(f"权重保存路径: {config['callbacks'][0]['init_args']['dirpath']}")
    print(f"总图片数量: {len(train_loader.dataset)}")
    
    trainer.fit(model, train_loader, val_loader)
    
    # 训练完成后冻结权重
    model.model.freeze_all()
    print("Backbone、STN和投影头权重已冻结")
    
    # 保存最终模型权重
    final_weights_path = os.path.join(
        config["weights"]["save_path"],
        "siamese_final_weights.pth"
    )
    torch.save(model.model.state_dict(), final_weights_path)
    print(f"最终模型权重已保存到: {final_weights_path}")
    
    # 单独保存STN权重
    stn_weights_path = config["weights"]["stn_weights_path"]
    torch.save(model.model.siamese_net.stn.state_dict(), stn_weights_path)
    print(f"STN权重已保存到: {stn_weights_path}")
    
    # 单独保存投影头权重
    projection_weights_path = config["weights"]["projection_weights_path"]
    torch.save(model.model.siamese_net.projection_head.state_dict(), projection_weights_path)
    print(f"投影头权重已保存到: {projection_weights_path}")


if __name__ == "__main__":
    # 检查必要的目录是否存在
    required_dirs = [
        "configs/pretrain",
        "D:/Projects/anomalib/weights/siamese_registration",
        "logs"
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    main()