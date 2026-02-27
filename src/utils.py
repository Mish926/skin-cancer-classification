"""
Utility Functions
-----------------
Shared helpers for training, evaluation, and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torchvision import transforms


# ── Checkpoint I/O ────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch: int, val_f1: float, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "val_f1":     val_f1,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, path: str, device: str = "cpu", optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val F1={checkpoint['val_f1']:.4f})")
    return checkpoint


# ── Image denormalization (for visualization) ─────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to a uint8 HWC numpy array."""
    img = tensor.cpu() * _STD + _MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ── Training curves ───────────────────────────────────────────────────────────
def plot_training_curves(history: dict, save_path: str):
    """
    Plots train/val loss and F1 side by side.
    history keys: train_loss, val_loss, train_f1, val_f1
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History — ResNet50 (HAM10000)", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", color="steelblue", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="tomato",    lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    # F1
    ax2.plot(epochs, history["train_f1"], label="Train", color="steelblue", lw=2)
    ax2.plot(epochs, history["val_f1"],   label="Val",   color="tomato",    lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Sample grid visualization ─────────────────────────────────────────────────
def plot_sample_grid(images, labels, preds=None, class_names=None, n=16, save_path=None):
    """
    Plots a grid of sample images with true (and optionally predicted) labels.
    images: list of CHW tensors (normalized)
    """
    cols = 4
    rows = (min(n, len(images)) + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(min(n, len(images))):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(denormalize(images[i]))
        true_label = class_names[labels[i]] if class_names else str(labels[i])
        title = f"True: {true_label}"
        color = "black"
        if preds is not None:
            pred_label = class_names[preds[i]] if class_names else str(preds[i])
            correct = (preds[i] == labels[i])
            title += f"\nPred: {pred_label}"
            color = "green" if correct else "red"
        ax.set_title(title, fontsize=7, color=color)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# ── Class distribution plot ───────────────────────────────────────────────────
def plot_class_distribution(labels, class_names: list, save_path: str = None):
    from collections import Counter
    counts = Counter(labels)
    sorted_items = sorted(counts.items())
    names  = [class_names[i] for i, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha="center", fontsize=9)
    ax.set_title("Class Distribution — HAM10000", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()
