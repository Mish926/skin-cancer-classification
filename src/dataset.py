"""
HAM10000 Dataset Loader
-----------------------
Handles data loading, preprocessing, augmentation, and class balancing
for the HAM10000 skin lesion dataset (10,015 dermatoscopic images, 7 classes).

Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ── Label mapping ──────────────────────────────────────────────────────────────
LESION_CLASSES = {
    "nv":   0,  # Melanocytic nevi
    "mel":  1,  # Melanoma
    "bkl":  2,  # Benign keratosis-like lesions
    "bcc":  3,  # Basal cell carcinoma
    "akiec":4,  # Actinic keratoses
    "vasc": 5,  # Vascular lesions
    "df":   6,  # Dermatofibroma
}
IDX_TO_CLASS = {v: k for k, v in LESION_CLASSES.items()}
CLASS_NAMES = [
    "Melanocytic nevi", "Melanoma", "Benign keratosis",
    "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions", "Dermatofibroma"
]

# ── Image stats (ImageNet defaults work well for transfer learning) ─────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Returns torchvision transforms for train / val / test splits.

    Train augmentations are chosen specifically for dermatoscopic images:
    - RandomHorizontalFlip + RandomVerticalFlip  (lesions have no canonical orientation)
    - ColorJitter                                (simulate varying dermoscope lighting)
    - RandomRotation                             (rotational invariance)
    - RandomResizedCrop                          (scale invariance, mimics zoom)
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Expected directory layout
    ─────────────────────────
    data/
      HAM10000_metadata.csv
      images/
        ISIC_0024306.jpg
        ISIC_0024307.jpg
        ...

    The metadata CSV must contain at minimum:
        image_id  |  dx  (lesion type abbreviation)

    Args:
        metadata_path: path to HAM10000_metadata.csv
        image_dir:     directory containing all .jpg images
        split:         'train', 'val', or 'test'
        transform:     optional override; if None uses get_transforms(split)
    """

    def __init__(
        self,
        metadata_path: str,
        image_dir: str,
        split: str = "train",
        transform=None,
    ):
        self.image_dir = image_dir
        self.transform = transform or get_transforms(split)

        df = pd.read_csv(metadata_path)

        # ── Deduplicate: keep one entry per lesion (some lesions have multiple images)
        df = df.drop_duplicates(subset="lesion_id", keep="first").reset_index(drop=True)

        # ── Train / val / test split (70 / 15 / 15) stratified by class
        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["dx"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["dx"], random_state=42
        )

        split_map = {"train": train_df, "val": val_df, "test": test_df}
        self.df = split_map[split].reset_index(drop=True)
        self.labels = self.df["dx"].map(LESION_CLASSES).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for WeightedRandomSampler."""
        counts = Counter(self.labels)
        total  = len(self.labels)
        weights = torch.zeros(len(LESION_CLASSES))
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / count
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for use with WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return class_weights[self.labels]


def build_dataloaders(
    metadata_path: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """
    Constructs train / val / test DataLoaders.

    The train loader uses WeightedRandomSampler to oversample minority classes
    (e.g. df, vasc) and bring per-class F1 scores up — critical for clinical utility.
    """
    datasets = {
        split: HAM10000Dataset(metadata_path, image_dir, split=split)
        for split in ("train", "val", "test")
    }

    # ── Weighted sampler for training only ─────────────────────────────────────
    sample_weights = datasets["train"].get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    print(f"Dataset sizes — train: {len(datasets['train'])}, "
          f"val: {len(datasets['val'])}, test: {len(datasets['test'])}")

    return loaders
