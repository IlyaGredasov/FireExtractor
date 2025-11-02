import os
from multiprocessing import freeze_support

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations import Compose
from albumentations import HorizontalFlip
from albumentations import Normalize
from albumentations import RandomBrightnessContrast
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

DATA_DIR = "fire_dataset"
IMG_SIZE = 512
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 30
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3  # 0=bg, 1=fire, 2=smoke


class FireSmokeDataset(Dataset):
    def __init__(self, root, split="train", augment=False, img_size=512):
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.images = [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.img_size = img_size
        self.augment = augment

        self.transform = Compose([
            Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            HorizontalFlip(p=0.5 if augment else 0.0),
            RandomBrightnessContrast(p=0.3 if augment else 0.0),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], is_check_shapes=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)
        base = os.path.splitext(name)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"].long()

        return img_t, mask_t


def train():
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(device)

    loss_fn = smp.losses.DiceLoss(mode="multiclass")
    metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_dataset = FireSmokeDataset(DATA_DIR, "train", augment=True)
    val_dataset = FireSmokeDataset(DATA_DIR, "valid", augment=False)
    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    best_metric = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        metric_indexes = []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                predictions = torch.argmax(logits, dim=1)
                iou = metric(predictions, masks)
                metric_indexes.append(iou.item())

        mean_metric = np.mean(metric_indexes)
        print(f"Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}, val metric_index={mean_metric:.4f}")
        scheduler.step()

        if mean_metric > best_metric:
            best_metric = mean_metric
            torch.save(model.state_dict(), "best_unet.pth")
            print("Saved best model")


if __name__ == "__main__":
    try:
        freeze_support()
    except Exception:
        pass
    train()
