import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

import kagglehub

from preprocess_data import collect_image_label_table

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dataset(repo_id: str, expected_subpaths=None) -> str:
    import os
    import kagglehub

    path = kagglehub.dataset_download(repo_id)
    print("Dataset path:", path)

    if expected_subpaths:
        for sp in expected_subpaths:
            full = os.path.join(path, sp)
            if not os.path.exists(full):
                raise RuntimeError(
                    f"Dataset downloaded, but expected path not found: {full}"
                )

    return path

class CalorieDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - image_path (str)
      - calories (float)
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        calories = float(row["calories"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        y = torch.tensor([calories], dtype=torch.float32)
        return img, y

@torch.no_grad()
def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.squeeze(1).cpu().numpy()
    target = target.squeeze(1).cpu().numpy()

    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target) ** 2)
    rmse = math.sqrt(mse)
    eps = 1e-8
    mape = np.mean(np.abs((pred - target) / (target + eps))) * 100.0
    within_10 = np.mean(np.abs(pred - target) <= 0.10 * (target + eps)) * 100.0

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE(%)": float(mape),
        "Within10%(%)": float(within_10),
    }

# CNN model
def build_cnn_regressor(backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    raise ValueError(f"Unknown backbone: {backbone}")

# train / eval
def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total = 0.0
    n = 0

    preds = []
    trues = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

        preds.append(pred.detach())
        trues.append(y.detach())

    avg_loss = total / max(n, 1)
    pred_cat = torch.cat(preds, dim=0)
    true_cat = torch.cat(trues, dim=0)

    # convert predictions back from log scale
    pred_cat = torch.expm1(pred_cat)
    true_cat = torch.expm1(true_cat)

    mets = regression_metrics(pred_cat, true_cat)
    return avg_loss, mets


def split_df(df: pd.DataFrame, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


@dataclass
class Config:
    kaggle_repo: str = "gillesokhin/nutrition5k-dataset"

    use_overhead: bool = True
    use_side_angles: bool = False
    max_images_total: int = 20000         
    max_images_per_dish: int = 2           

    # training config
    image_size: int = 224
    backbone: str = "resnet18"             
    pretrained: bool = True

    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    seed: int = 42

    # output
    out_dir: str = "runs/cnn_baseline"
    ckpt_name: str = "best_model.pt"
    save_labels_csv: bool = True


def main():
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset_path = ensure_dataset(
        cfg.kaggle_repo,
        expected_subpaths=["dish_nutrition_values.csv", "imagery"]
    )
    print("Dataset path:", dataset_path)

    # build (image_path, calories) table, mapping calorie value to dish
    labels_csv = os.path.join("data_cache", "labels_built.csv")
    os.makedirs("data_cache", exist_ok=True)

    if os.path.exists(labels_csv):
        print(f"Using existing labels file: {labels_csv}")
        df = pd.read_csv(labels_csv)
    else:
        print("labels_built.csv not found. Building it from dataset...")
        df = collect_image_label_table(
            dataset_path,
            use_overhead=cfg.use_overhead,
            use_side_angles=cfg.use_side_angles,
            max_images_total=cfg.max_images_total,
            max_images_per_dish=cfg.max_images_per_dish,
            seed=cfg.seed
    )
        
    # log-transform calories 
    df["calories"] = np.log1p(df["calories"])

    df.to_csv(labels_csv, index=False)
    print(f"Saved labels to: {labels_csv}")

    print(f"Built dataset table with {len(df)} rows")
    print(df.head())

    train_df, val_df, test_df = split_df(df, seed=cfg.seed, train_frac=0.8, val_frac=0.1)
    print(f"Split sizes -> train: {len(train_df)} val: {len(val_df)} test: {len(test_df)}")

    # training transforms (augmentation)
    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # validation/test transforms (no augmentation, only resize + normaliza)
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # datasets and loaders
    train_ds = CalorieDataset(train_df, transform=train_tf)
    val_ds = CalorieDataset(val_df, transform=eval_tf)
    test_ds = CalorieDataset(test_df, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    model = build_cnn_regressor(cfg.backbone, pretrained=cfg.pretrained).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")

    # training loop
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mets = evaluate(model, val_loader, loss_fn, device)

        dt = time.time() - t0
        print(f"\nEpoch {epoch}/{cfg.epochs}  ({dt:.1f}s)")
        print(f"  Train MSE: {train_loss:.4f}")
        print(f"  Val   MSE: {val_loss:.4f}")
        print(f"  Val metrics: {val_mets}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "val_loss": val_loss,
                    "val_metrics": val_mets,
                },
                ckpt_path
            )
            print(f"  ✓ Saved best checkpoint to {ckpt_path}")

    # test evaluation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_mets = evaluate(model, test_loader, loss_fn, device)
    print("\nTest Results")
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test metrics: {test_mets}")

    with open(os.path.join(cfg.out_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test MSE: {test_loss:.6f}\n")
        for k, v in test_mets.items():
            f.write(f"{k}: {v:.6f}\n")

    print("Saved test metrics to:", os.path.join(cfg.out_dir, "test_metrics.txt"))


if __name__ == "__main__":
    main()