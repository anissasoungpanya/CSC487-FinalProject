import math
from typing import Dict, Tuple
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import torch

from vit_helpers.class_defs import NutritionDataset

@torch.no_grad()
def regression_metrics(pred: torch.Tensor, target: torch.Tensor, percent_correct_threshold=0.1) -> Dict[str, float]:
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()

    # Standard metrics (unaffected by the zero-division bug)
    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target) ** 2)
    rmse = math.sqrt(mse)
    
    denominator = (np.abs(pred) + np.abs(target)) / 2.0
    smape = np.mean(np.abs(pred - target) / (denominator + 1e-8)) * 100.0
    
    valid_mask = target >= 1.0

    # used a slightly adjusted mape since was getting crazy million-level percentage errors, presumably because it was very small caloric values
    # also added smape above
    if np.any(valid_mask):
        safe_pred = pred[valid_mask]
        safe_target = target[valid_mask]
        
        mape = np.mean(np.abs((safe_pred - safe_target) / safe_target)) * 100.0
        within = np.mean(np.abs(safe_pred - safe_target) <= percent_correct_threshold * safe_target) * 100.0
    else:
        mape = 0.0
        within = 0.0

    return {
        "Cal_MAE": float(mae),
        "Cal_MSE": float(mse),
        "Cal_RMSE": float(rmse),
        "Cal_SMAPE(%)": float(smape),
        "Cal_MAPE(%)": float(mape),
        f"Cal_Within{percent_correct_threshold*100}%(%)": float(within),
    }

def train_one_epoch_dual(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        pred_aux, pred_final = model(x)
        
        loss = loss_fn(pred_aux, pred_final, y)
        
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate_dual(model, loader, loss_fn, device, percent_correct_cutoff) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n = 0

    preds = []
    trues = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True) 

        pred_aux, pred_final = model(x)
        
        loss = loss_fn(pred_aux, pred_final, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        preds.append(pred_final.detach())
        
        true_calories = y[:, 0].unsqueeze(1)
        trues.append(true_calories.detach())

    avg_loss = total_loss / max(n, 1)
    
    pred_cat = torch.cat(preds, dim=0)
    true_cat = torch.cat(trues, dim=0)

    pred_cat = torch.expm1(pred_cat)
    true_cat = torch.expm1(true_cat)
    
    mets = regression_metrics(pred_cat, true_cat, percent_correct_cutoff)
    return avg_loss, mets

def split_df(df: pd.DataFrame, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    unique_dishes = df['dish_id'].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    
    n = len(unique_dishes)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    
    train_ids = set(unique_dishes[:n_train])
    val_ids = set(unique_dishes[n_train:n_train + n_val])
    test_ids = set(unique_dishes[n_train + n_val:])
    
    train_df = df[df['dish_id'].isin(train_ids)].copy()
    val_df = df[df['dish_id'].isin(val_ids)].copy()
    test_df = df[df['dish_id'].isin(test_ids)].copy()
    
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print(f"Split sizes (Images) -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def gen_transforms(cfg):
    mean = [0.5, 0.5, 0.5] # according to gemini, resnet used the prior normalization values, but the pretrained vit likely used 0.5? seems to train towards better results this way, so decided to keep it
    std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform

def gen_nutrition_dataloaders(train_df, val_df, test_df, train_transform, eval_transform, cfg):
    train_ds = NutritionDataset(train_df, transform=train_transform)
    val_ds = NutritionDataset(val_df, transform=eval_transform)
    test_ds = NutritionDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=cfg.num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader