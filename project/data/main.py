"""Entry point for object detection training.

This script builds data loaders, creates the model, selects a device,
and starts the training loop.
"""

from args import get_args
from dataset import ObjDetectionDataset, preview_train_batch
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model
from augmentations import build_train_transforms, build_val_transforms



def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()
    print(args)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    csv_dir = args.csv_dir
    if not os.path.isabs(csv_dir) and not os.path.exists(csv_dir):
        csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_dir)

    train_df = pd.read_csv(os.path.join(csv_dir, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(csv_dir, "val_df.csv"))

    image_size = 640
    train_dataset = ObjDetectionDataset(train_df, base_dir=base_dir, transforms=build_train_transforms(image_size))
    val_dataset = ObjDetectionDataset(val_df, base_dir=base_dir, transforms=build_val_transforms(image_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())

    preview_train_batch(val_loader)

    model = build_model(args.backbone)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_model(model, train_loader, val_loader, device)

    


if __name__ == "__main__":
    main()