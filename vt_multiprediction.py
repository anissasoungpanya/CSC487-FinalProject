from train_vt import VisionTransformer, gen_dataloaders, gen_transforms
from torch import nn


def main():
    model = VisionTransformer(num_classes=5)
    model.classifier = nn.Sequential(
        nn.Linear(), 
        nn.SiLU(), 
        nn.Dropout(0.5), 
        nn.Linear(), 
        nn.SiLU(), 
        nn.Linear(5), 
        nn.SiLU(), 
        nn.Dropout(0.5),
        nn.Linear(, 5)
    )
    pass

if __name__ == "__main__":
    main()