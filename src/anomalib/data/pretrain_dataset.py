"""改进的预训练数据集实现

实现SimSiam风格的Siamese配准网络预训练数据集，解决特征塌陷问题。
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Union, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class PretrainDataset(Dataset):
    """改进的Siamese配准网络预训练数据集
    
    按类别构建索引，确保采样同一类别的不同图片对，避免特征塌陷。
    """
    
    def __init__(
        self,
        root: str,
        category: Union[str, List[str]],
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        augmentations: Optional[Callable] = None
    ):
        """
        Args:
            root: 数据集根目录
            category: 类别名称或类别列表
            split: 数据分割 ('train' 或 'test')
            image_size: 图片尺寸
            transform: 基础图片变换
            augmentations: 数据增强变换
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        # 处理单个类别或多个类别
        if isinstance(category, str):
            categories = [category]
        else:
            categories = category
        self.categories = categories
        
        # 按类别构建图片路径索引
        self.cat2paths: Dict[str, List[Path]] = {}
        self.cat_indices: Dict[str, List[int]] = {}
        
        total_idx = 0
        for cat in categories:
            image_dir = self.root / cat / split / "good"
            if not image_dir.exists():
                raise ValueError(f"图片目录不存在: {image_dir}")
            
            # 获取当前类别的所有图片文件
            cat_paths = sorted([
                p for p in image_dir.iterdir() 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ])
            
            if len(cat_paths) == 0:
                raise ValueError(f"类别 {cat} 中没有找到图片文件")
            
            self.cat2paths[cat] = cat_paths
            self.cat_indices[cat] = list(range(total_idx, total_idx + len(cat_paths)))
            total_idx += len(cat_paths)
        
        # 构建全局索引到类别和局部索引的映射
        self.global_to_cat = {}
        self.global_to_local = {}
        
        for cat, indices in self.cat_indices.items():
            for global_idx, local_idx in enumerate(indices):
                self.global_to_cat[local_idx] = cat
                self.global_to_local[local_idx] = global_idx
        
        self.total_samples = total_idx
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # 数据增强（用于生成不同的增广版本）
        if augmentations is None:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        else:
            self.augmentations = augmentations
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """获取同一类别的两张不同图片的增广版本"""
        # 获取类别
        cat = self.global_to_cat[idx]
        
        # 使用sample_pair方法获取同一类别的不同图片对
        img1, img2, sampled_cat = self.sample_pair(category=cat)
        
        # 确保采样类别与索引类别一致
        assert sampled_cat == cat, f"采样类别不匹配: {sampled_cat} != {cat}"
        
        return img1, img2, cat
    
    def sample_pair(self, category: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """采样同一类别的两张不同图片的增广版本"""
        # 如果未指定类别，随机选择一个类别
        if category is None:
            cat = random.choice(self.categories)
        else:
            cat = category
        
        # 获取该类别的所有图片路径
        cat_paths = self.cat2paths[cat]
        
        # 如果该类别只有一张图，退化为两个增广版本
        if len(cat_paths) < 2:
            img_path = cat_paths[0]
            image = Image.open(img_path).convert('RGB')
            
            # 生成两个不同的增广版本
            aug_image1 = self.augmentations(image)
            aug_image1 = self.transform(aug_image1)
            
            aug_image2 = self.augmentations(image)
            aug_image2 = self.transform(aug_image2)
            
            return aug_image1, aug_image2, cat
        else:
            # 随机选择两张不同的图片
            img_path1, img_path2 = random.sample(cat_paths, 2)
            
            # 加载并增广第一张图片
            image1 = Image.open(img_path1).convert('RGB')
            aug_image1 = self.augmentations(image1)
            aug_image1 = self.transform(aug_image1)
            
            # 加载并增广第二张图片
            image2 = Image.open(img_path2).convert('RGB')
            aug_image2 = self.augmentations(image2)
            aug_image2 = self.transform(aug_image2)
            
            return aug_image1, aug_image2, cat


class SiameseDataLoader:
    """Siamese网络数据加载器
    
    支持从PretrainDataset中采样不同类别的图片对。
    """
    
    def __init__(
        self,
        dataset: PretrainDataset,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
        if self.shuffle:
            import random
            random.shuffle(self.indices)
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_idx >= len(self.indices):
            if self.shuffle:
                import random
                random.shuffle(self.indices)
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # 为每个样本生成一对增广图片
        batch1 = []
        batch2 = []
        
        for idx in batch_indices:
            img1, img2 = self.dataset[idx]
            batch1.append(img1)
            batch2.append(img2)
        
        self.current_idx = end_idx
        
        return torch.stack(batch1), torch.stack(batch2)
    
    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def create_pretrain_dataloader(
    root: str,
    category: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4
):
    """创建预训练数据加载器"""
    dataset = PretrainDataset(
        root=root,
        category=category,
        split="train",
        image_size=image_size
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )