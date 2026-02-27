"""
Training Pipeline
-----------------
Two-phase training strategy:
  Phase 1 (warm-up, ~5 epochs): Frozen backbone, train classifier head only.
                                  High LR safe because only new weights update.
  Phase 2 (fine-tune, remaining): Unfreeze all layers, discriminative LRs
                                   (lower for backbone, higher for head).

Key components:
  - Focal Loss: downweights easy negatives, focuses on hard minority classes
  - CosineAnnealingWarmRestarts scheduler
  - Early stopping on val loss (patience=7)
  - MLflow experiment tracking
"""

import os
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import f1_score

from src.dataset import build_dataloaders, LESION_CLASSES
from src.model import build_model
from src.utils import save_checkpoint, load_checkpoint, plot_training_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — addresses class imbalance by down-weighting
    well-classified examples. gamma=2 is the standard recommendation.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Training / Validation Loops ───────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


# ── Main Training Function ────────────────────────────────────────────────────
def train(config: dict):
    # Select best available device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Training on: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    loaders = build_dataloaders(
        metadata_path=config["metadata_path"],
        image_dir=config["image_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(freeze_layers=2, device=device)

    # ── Class-weighted Focal Loss ────────────────────────────────────────────
    class_weights = loaders["train"].dataset.get_class_weights().to(device)
    criterion     = FocalLoss(gamma=2.0, weight=class_weights)

    # ── Phase 1 optimizer (head only) ───────────────────────────────────────
    head_params = list(model.classifier.parameters())
    optimizer   = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["warmup_epochs"], T_mult=1
    )
    early_stop  = EarlyStopping(patience=config["patience"])

    best_val_f1 = 0.0
    history     = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    with mlflow.start_run(run_name="resnet50-skin-cancer"):
        mlflow.log_params({k: v for k, v in config.items()
                           if not isinstance(v, (dict, list))})

        for epoch in range(1, config["epochs"] + 1):

            # ── Switch to full fine-tune after warm-up ───────────────────────
            if epoch == config["warmup_epochs"] + 1:
                logger.info("Warm-up complete — unfreezing full backbone.")
                model.unfreeze_all()
                optimizer = optim.AdamW([
                    {"params": model.feature_extractor.parameters(), "lr": 1e-5},
                    {"params": model.classifier.parameters(),         "lr": 1e-4},
                ], weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=config["epochs"] - config["warmup_epochs"]
                )

            t0 = time.time()
            train_loss, train_acc, train_f1 = run_epoch(
                model, loaders["train"], criterion, optimizer, device, train=True
            )
            val_loss, val_acc, val_f1 = run_epoch(
                model, loaders["val"], criterion, optimizer, device, train=False
            )
            scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{config['epochs']} | "
                f"train loss {train_loss:.4f} f1 {train_f1:.4f} | "
                f"val loss {val_loss:.4f} f1 {val_f1:.4f} | "
                f"{elapsed:.1f}s"
            )

            # ── MLflow logging ───────────────────────────────────────────────
            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "train_f1":   train_f1,   "val_f1":   val_f1,
                "train_acc":  train_acc,  "val_acc":  val_acc,
            }, step=epoch)

            for key, val in zip(
                ["train_loss", "val_loss", "train_f1", "val_f1"],
                [train_loss,   val_loss,   train_f1,   val_f1]
            ):
                history[key].append(val)

            # ── Checkpoint best model ────────────────────────────────────────
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(model, optimizer, epoch, val_f1,
                                path=config["checkpoint_path"])
                mlflow.pytorch.log_model(model, "best_model")
                logger.info(f"  ✓ New best val F1: {best_val_f1:.4f} — checkpoint saved")

            if early_stop(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # ── Plot and save training curves ────────────────────────────────────
        plot_training_curves(history, save_path="results/figures/training_curves.png")
        mlflow.log_artifact("results/figures/training_curves.png")

        logger.info(f"Training complete. Best val macro-F1: {best_val_f1:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin lesion classifier")
    parser.add_argument("--metadata_path",   type=str, default="data/HAM10000_metadata.csv")
    parser.add_argument("--image_dir",       type=str, default="data/images")
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--epochs",          type=int, default=30)
    parser.add_argument("--warmup_epochs",   type=int, default=5)
    parser.add_argument("--patience",        type=int, default=7)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--checkpoint_path", type=str, default="results/best_model.pth")
    args = parser.parse_args()

    train(vars(args))
