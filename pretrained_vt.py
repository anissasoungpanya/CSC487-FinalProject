import time
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import ViTModel

from train_cnn import (evaluate, set_seed, ensure_dataset, CalorieDataset, 
                       regression_metrics, Config, split_df, train_one_epoch)
from preprocess_data import *

class PretrainedViTForRegression(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', num_classes=1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        
        # for if only want to fine-tune classifier portion
        # for param in self.vit.parameters():
        #     param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        return self.classifier(cls_output)

def gen_dataloaders(
        train_dataframe, validate_dataframe, test_dataframe, 
        train_transform, evaluation_transform, conf: Config):
    
    train_ds = CalorieDataset(train_dataframe, transform=train_transform)
    val_ds = CalorieDataset(validate_dataframe, transform=evaluation_transform)
    test_ds = CalorieDataset(test_dataframe, transform=evaluation_transform)

    train_loader = DataLoader(
        train_ds, batch_size=conf.batch_size, shuffle=True,
        num_workers=conf.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=conf.batch_size, shuffle=False,
        num_workers=conf.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=conf.batch_size, shuffle=False,
        num_workers=conf.num_workers, pin_memory=True
    )

    return train_loader, test_loader, val_loader

def gen_transforms(conf: Config):
    # standard ImageNet means/stds are expected by the pre-trained HF model
    train_tf = transforms.Compose([
        transforms.Resize((conf.image_size, conf.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(conf.image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((conf.image_size, conf.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def main():
    cfg = Config(out_dir="runs/vit_hf_classifier_finetuned_wLog")
    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # ensure dataset is downloaded
    dataset_path = ensure_dataset(
        cfg.kaggle_repo,
        expected_subpaths=["dish_nutrition_values.csv", "imagery"]
    )
    print("Dataset path:", dataset_path)

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
            seed=cfg.seed)
        df.to_csv(labels_csv, index=False)
        print(f"Saved labels to: {labels_csv}")

    df["calories"] = np.log1p(df["calories"])

    print(f"Built dataset table with {len(df)} rows")
    
    # train/val/test split
    train_df, val_df, test_df = split_df(df, seed=cfg.seed, train_frac=0.8, val_frac=0.1)
    
    train_transform, eval_transform = gen_transforms(cfg)
    train_loader, test_loader, val_loader = gen_dataloaders(
        train_df, val_df, test_df, train_transform, eval_transform, cfg
    )
    
    model = PretrainedViTForRegression(model_name='google/vit-base-patch16-224-in21k', num_classes=1)
    
    model = model.to(device)
    
    loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")

    # training loop
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        print(f"\nStarting Epoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mets = evaluate(model, val_loader, loss_fn, device)

        dt = time.time() - t0
        print(f"Epoch {epoch}/{cfg.epochs} Summary ({dt:.1f}s)")
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
    print("\n--- Testing Best Model ---")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_mets = evaluate(model, test_loader, loss_fn, device)
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test metrics: {test_mets}")

    with open(os.path.join(cfg.out_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test MSE: {test_loss:.6f}\n")
        for k, v in test_mets.items():
            f.write(f"{k}: {v:.6f}\n")

    print("Saved test metrics to:", os.path.join(cfg.out_dir, "test_metrics.txt"))

if __name__ == "__main__":
    main()