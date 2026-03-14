from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
from torch import nn
from transformers import ViTModel

class NutritionDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - image_path (str)
      - calories, mass, fat, protein, carbohydrates (float)
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.targets = ["calories", "mass", "fat", "protein", "carbohydrates"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        
        # Grab all 5 target values
        y_vals = [float(row[t]) for t in self.targets]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Output a tensor of size 5
        y = torch.tensor(y_vals, dtype=torch.float32)
        return img, y
    
class DualNutritionLoss(nn.Module):
    def __init__(self, final_weight: float = 0.7, aux_weight: float = 0.3):
        super().__init__()
        self.final_weight = final_weight
        self.aux_weight = aux_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_aux: torch.Tensor, pred_final: torch.Tensor, target_all: torch.Tensor):
        loss_aux = self.mse(pred_aux, target_all)
        
        target_cal = target_all[:, 0].unsqueeze(1) 
        loss_final = self.mse(pred_final, target_cal)
        
        total_loss = (self.aux_weight * loss_aux) + (self.final_weight * loss_final)
        
        return total_loss

# super similar to the model in pretrained_vt, probably could have made some overlap? but didn't seem worth the effort.
class PretrainedViTWithDualClassifier(nn.Module): 
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', num_broad=5, num_narrow=1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)

        for param in self.vit.parameters():
            param.requires_grad = False

        self.broad_classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Linear(self.vit.config.hidden_size, 128), 
            nn.SiLU(), 
            nn.Dropout(0.2), 
            nn.Linear(128, num_broad), # predict all 5 values we have access to in the dataset (calorie, fat, weight, carbs, protien)
        )
        self.narrow_classifier = nn.Sequential( # will take the five outputs, alongside the CLS output
            nn.Linear(num_broad + self.vit.config.hidden_size, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_narrow)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        
        cls_output = outputs.last_hidden_state[:, 0, :]

        broad_outputs = self.broad_classifier(cls_output)
        narrow_output = self.narrow_classifier(torch.cat([broad_outputs, cls_output], dim=-1))        

        return broad_outputs, narrow_output