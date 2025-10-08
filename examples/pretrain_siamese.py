"""RegAD training script adapted for Anomalib examples.

This script follows the training procedure described in the official
`MediaBrain-SJTU/RegAD` repository.  It pre-trains the Siamese registration
network (STN backbone + encoder + predictor) using the symmetric cosine loss
on the MVTec dataset and stores the learned checkpoints together with sampled
support sets.  The implementation mirrors the reference code but is updated to
use the modules integrated under ``anomalib.models.image.regmm``.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from anomalib.models.image.regmm.siamese import CosLoss, Encoder, Predictor, N_PARAMS, stn_net

BACKBONE_NAME = "wide_resnet50_2"

try:
    import kornia as K  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Kornia is required for RegAD training") from exc


# ---------------------------------------------------------------------------
# Dataset utilities (adapted from RegAD's ``datasets/mvtec.py``)


class FSADDatasetTrain(Dataset):
    """Loads normal training images for a specific MVTec category."""

    def __init__(self, data_path: str, class_name: str, resize: int) -> None:
        super().__init__()
        self.paths = self._collect_paths(data_path, class_name, split="train")
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=Image.BILINEAR),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _collect_paths(data_path: str, class_name: str, split: str) -> List[Path]:
        base = Path(data_path) / class_name / split
        # Only the "good" folder is used for few-shot training
        candidates = list((base / "good").rglob("*.png")) + list((base / "good").rglob("*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"No training images found in {base / 'good'}")
        candidates.sort()
        return candidates

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Auxiliary functions copied from RegAD utils


def embedding_concat(x: Tensor, y: Tensor, use_cuda: bool) -> Tensor:
    device = torch.device("cuda" if use_cuda else "cpu")
    bsz, c1, h1, w1 = x.size()
    _, c2, h2, w2 = y.size()
    stride = int(h1 / h2)
    unfolded = F.unfold(x, kernel_size=stride, dilation=1, stride=stride)
    unfolded = unfolded.view(bsz, c1, -1, h2, w2)
    z = torch.zeros(bsz, c1 + c2, unfolded.size(2), h2, w2, device=device)
    for i in range(unfolded.size(2)):
        z[:, :, i, :, :] = torch.cat((unfolded[:, :, i, :, :], y), 1)
    z = z.view(bsz, -1, h2 * w2)
    z = F.fold(z, kernel_size=stride, output_size=(h1, w1), stride=stride)
    return z


def rot_img(x: Tensor, theta: float) -> Tensor:
    dtype = torch.float32
    rot_mat = get_rot_mat(theta)[None, ...].to(dtype=dtype, device=x.device).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


def translation_img(x: Tensor, a: float, b: float) -> Tensor:
    dtype = torch.float32
    mat = get_translation_mat(a, b)[None, ...].to(dtype=dtype, device=x.device).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(mat, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


def hflip_img(x: Tensor) -> Tensor:
    return K.geometry.transform.hflip(x)


def rot90_img(x: Tensor, k: int) -> Tensor:
    degreesarr = [0.0, 90.0, 180.0, 270.0, 360.0]
    degrees = torch.tensor(degreesarr[k], device=x.device)
    return K.geometry.transform.rotate(x, angle=degrees, padding_mode="reflection")


def grey_img(x: Tensor) -> Tensor:
    x = K.color.rgb_to_grayscale(x)
    return x.repeat(1, 3, 1, 1)


def get_rot_mat(theta: float) -> Tensor:
    theta_t = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta_t), -torch.sin(theta_t), 0], [torch.sin(theta_t), torch.cos(theta_t), 0]])


def get_translation_mat(a: float, b: float) -> Tensor:
    return torch.tensor([[1, 0, a], [0, 1, b]])


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Training utilities


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_support_set(dataset: FSADDatasetTrain, shot: int, inferences: int, device: torch.device) -> List[Tensor]:
    """Sample support sets following RegAD's augmentation recipe."""

    total_required = shot * inferences
    indices = torch.randperm(len(dataset))
    if indices.numel() < total_required:
        repeat = (total_required + indices.numel() - 1) // indices.numel()
        indices = indices.repeat(repeat)
    support_collections: List[Tensor] = []
    for i in range(inferences):
        chosen = indices[i * shot : (i + 1) * shot]
        imgs = torch.stack([dataset[j] for j in chosen]).to(device)
        support = augment_support_images(imgs)
        support_collections.append(support.cpu())
    return support_collections


def augment_support_images(support_img: Tensor) -> Tensor:
    augment_support_img = support_img
    device = support_img.device
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16, np.pi / 4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    perm = torch.randperm(augment_support_img.size(0), device=device)
    return augment_support_img[perm]


def train_epoch(
    args: argparse.Namespace,
    models: Tuple[nn.Module, nn.Module, nn.Module],
    optimizer: SGD,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    stn_model, encoder, predictor = models
    stn_model.train()
    encoder.train()
    predictor.train()

    losses = AverageMeter()
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        support_idx = torch.randperm(batch.size(0), device=device)[: args.shot]
        support_img = batch[support_idx]
        support_aug = augment_support_images(support_img)
        support_feat = stn_model(support_aug)
        support_feat = support_feat.mean(dim=0, keepdim=True)

        query_feat = stn_model(batch)
        z1 = encoder(query_feat)
        z2 = encoder(support_feat)
        p1 = predictor(z1)
        p2 = predictor(z2)

        loss = 0.5 * (CosLoss(p1, z2, mean=True) + CosLoss(p2, z1, mean=True))
        loss.backward()
        optimizer.step()

        losses.update(float(loss.item()), batch.size(0))

    return losses.avg


def save_checkpoint(
    args: argparse.Namespace,
    stn_model: nn.Module,
    encoder: nn.Module,
    predictor: nn.Module,
    optimizer: SGD,
) -> Path:
    save_dir = Path("save_checkpoints") / args.stn_mode / str(args.shot) / args.obj
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{args.obj}_{args.shot}_{args.stn_mode}_model.pt"
    torch.save({"STN": stn_model.state_dict(), "ENC": encoder.state_dict(), "PRED": predictor.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
    return ckpt_path


def save_support_sets(args: argparse.Namespace, support_sets: Sequence[Tensor]) -> Path:
    save_dir = Path("support_set") / args.obj
    save_dir.mkdir(parents=True, exist_ok=True)
    support_path = save_dir / f"{args.shot}_{args.inferences}.pt"
    torch.save(support_sets, support_path)
    return support_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RegAD training on MVTec")
    parser.add_argument("--obj", type=str, default="hazelnut")
    parser.add_argument("--data_type", type=str, default="mvtec")
    parser.add_argument("--data_path", type=str, default="./MVTec/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--inferences", type=int, default=10)
    parser.add_argument(
        "--stn_mode",
        type=str,
        default="rotation_scale",
        choices=list(N_PARAMS.keys()),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = FSADDatasetTrain(args.data_path, class_name=args.obj, resize=args.img_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=True,
    )

    print(f"Initializing STN backbone: {BACKBONE_NAME}")
    stn_model = stn_net(args.stn_mode, pretrained=True).to(device)
    encoder = Encoder().to(device)
    predictor = Predictor().to(device)

    params = list(stn_model.parameters()) + list(encoder.parameters()) + list(predictor.parameters())
    optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    log_dir = Path("logs") / args.obj
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{time.strftime('%Y%m%d-%H%M%S')}.log"
    with log_path.open("w") as log_file:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = train_epoch(args, (stn_model, encoder, predictor), optimizer, train_loader, device)
            msg = f"Epoch [{epoch}/{args.epochs}] - loss: {epoch_loss:.6f}"
            print(msg)
            log_file.write(msg + "\n")

    ckpt_path = save_checkpoint(args, stn_model, encoder, predictor, optimizer)
    print(f"Checkpoint saved to {ckpt_path}")

    support_sets = create_support_set(train_dataset, shot=args.shot, inferences=args.inferences, device=device)
    support_path = save_support_sets(args, support_sets)
    print(f"Support sets saved to {support_path}")


if __name__ == "__main__":
    main()

