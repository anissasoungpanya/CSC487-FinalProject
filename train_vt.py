import time

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from train_cnn import evaluate, set_seed, ensure_dataset, CalorieDataset, regression_metrics, Config, split_df, train_one_epoch
import os
from preprocess_data import *
from torchvision import transforms

# https://arxiv.org/pdf/2010.11929

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        
        # vision transformers need sequences just like normal transformers - this conv layer
        # converts a single image into sequences of "patches", and then we can flatten those into 1-d 
        # embeddings thru the embed_dim number of filters. 
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # Input shape: (Batch, Channels, Height, Width)
        x = self.proj(x) # (Batch, embed, H/patch, W/patch)
        x = x.flatten(2) # (Batch, embed, num_patches)
        x = x.transpose(1, 2)  # Shape: (Batch, num_patches, embed)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, 
                 num_heads=12, num_layers=12, num_classes=4): 
        super().__init__()
        
        self.embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        # vision transformers "output" data in a couple different ways.. we could average out the
        # final output of all attention heads, or use an additional sequence patch (this cls token) as a kind of
        # "sum" patch that gains features through each layer + attention heads. decided to use cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.embedding(x)
        
        # add the cls token (need to append cls token to each sequence in the batch)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.transformer(x)
        cls_output = x[:, 0]
        
        return self.classifier(cls_output)
    
    def override_classifier(self, to):
        self.classifier = to

def gen_dataloaders(
        train_dataframe, 
        validate_dataframe, 
        test_dataframe, 
        train_transform, 
        evaluation_transform, 
        conf: Config):
    # datasets and loaders
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
    # training transforms (augmentation)
    train_tf = transforms.Compose([
        transforms.Resize((conf.image_size, conf.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(conf.image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # validation/test transforms (no augmentation, only resize + normaliza)
    eval_tf = transforms.Compose([
        transforms.Resize((conf.image_size, conf.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def main():
    cfg = Config(out_dir="runs/vit_mod.txt")
    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ensure dataset is downloaded
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
    df.to_csv(labels_csv, index=False)
    print(f"Saved labels to: {labels_csv}")

    print(f"Built dataset table with {len(df)} rows")
    print(df.head())

    # train/val/test split
    train_df, val_df, test_df = split_df(df, seed=cfg.seed, train_frac=0.8, val_frac=0.1)
    print(f"Split sizes -> train: {len(train_df)} val: {len(val_df)} test: {len(test_df)}")

    train_transform, eval_transform = gen_transforms(cfg)
    train_loader, val_loader, test_loader = gen_dataloaders(train_df, val_df, test_df, train_transform, eval_transform, cfg)
    
    model = VisionTransformer() # todo - set up model params
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
    
    