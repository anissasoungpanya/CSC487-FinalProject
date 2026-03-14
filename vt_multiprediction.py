import os
import time

import torch
from train_cnn import Config, set_seed
from vit_helpers.class_defs import DualNutritionLoss, PretrainedViTWithDualClassifier
from vit_helpers.multi_helpers import evaluate_dual, gen_nutrition_dataloaders, gen_transforms, split_df, train_one_epoch_dual
from vit_helpers.preprocess import ensure_dataset, get_df_from_labels

def train_loop(cfg, model, train_loader, val_loader, loss_fn, optimizer, device, ckpt_path):
    # Track the best MAE instead of the best Loss
    best_mae = float("inf") 
    
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        print(f"\nStarting Epoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch_dual(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mets = evaluate_dual(model, val_loader, loss_fn, device)

        dt = time.time() - t0
        print(f"Epoch {epoch}/{cfg.epochs} Summary ({dt:.1f}s)")
        print(f"  Train MSE (Log Scale): {train_loss:.4f}")
        print(f"  Val Total Loss: {val_loss:.4f}")
        print(f"  Val metrics: {val_mets}")

        # saving best model not off the loss but rather one specific metric. decided to go w/ mae
        # rather than rmse bc imo, getting double the error off means that the model is twice as bad
        current_mae = val_mets['Cal_MAE']
        
        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "val_loss": val_loss,
                    "val_metrics": val_mets,
                },
                ckpt_path
            )
            print(f"  ✓ Saved best checkpoint to {ckpt_path} (MAE: {current_mae:.2f})")

def test_eval(cfg, model, test_loader, loss_fn, device, ckpt_path, percent_correct, output_file):
    print("\n--- Testing Best Model ---")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_mets = evaluate_dual(model, test_loader, loss_fn, device, percent_correct)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test metrics: {test_mets}")

    with open(os.path.join(cfg.out_dir, output_file), "w") as f:
        f.write(f"Test loss: {test_loss:.6f}\n")
        for k, v in test_mets.items():
            f.write(f"{k}: {v:.6f}\n")

    print("Saved test metrics to:", os.path.join(cfg.out_dir, "test_metrics.txt"))

def main():
    model = PretrainedViTWithDualClassifier()

    cfg = Config(out_dir="runs/vit_multi_class", epochs=20)
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

    model.to(device)

    # ensure dataset is downloaded
    dataset_path = ensure_dataset(
        cfg.kaggle_repo,
        expected_subpaths=["dish_nutrition_values.csv", "imagery"]
    )
    print("Dataset path:", dataset_path)

    df = get_df_from_labels(dataset_path, cfg)
    print(f"Built dataset table with {len(df)} rows")
    
    # train/val/test split
    train_df, val_df, test_df = split_df(df, seed=cfg.seed, train_frac=0.8, val_frac=0.1)
    
    train_transform, eval_transform = gen_transforms(cfg)
    train_loader, val_loader, test_loader = gen_nutrition_dataloaders(
        train_df, val_df, test_df, train_transform, eval_transform, cfg
    )
        
    loss_fn = DualNutritionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # train_loop(
    #     cfg, 
    #     model, 
    #     train_loader, 
    #     val_loader, 
    #     loss_fn ,
    #     optimizer, 
    #     device, 
    #     ckpt_path
    # ) 

    print("\n fine tuning now; unfreezing all of the transformer weights and lowering LR")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best Phase 1 model with MAE: {ckpt['val_metrics']['Cal_MAE']:.2f}")

    for param in model.vit.parameters():
        param.requires_grad = True
        
    fine_tune_lr = cfg.lr / 10.0 # or manually set to 1e-5 / 2e-5
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=cfg.weight_decay)
    
    ft_ckpt_path = os.path.join(cfg.out_dir, "best_model_finetuned.pt")
    
    # train_loop(cfg, model, train_loader, val_loader, loss_fn, optimizer_ft, device, ft_ckpt_path)

    test_eval(cfg, model, test_loader, loss_fn, device, ft_ckpt_path, 0.50, "test_eval_50percent")

if __name__ == "__main__":
    main()