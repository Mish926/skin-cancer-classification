"""
Evaluation & Inference
----------------------
Runs the trained model on the held-out test set and produces:
  - Overall accuracy, macro/weighted F1, AUC-ROC (one-vs-rest)
  - Per-class precision, recall, F1
  - Confusion matrix heatmap  →  results/figures/confusion_matrix.png
  - ROC curves per class       →  results/figures/roc_curves.png
  - Misclassified sample grid  →  results/figures/misclassified.png
  - Full metrics JSON          →  results/test_metrics.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
)

from src.dataset import build_dataloaders, CLASS_NAMES, IDX_TO_CLASS
from src.model import build_model, NUM_CLASSES
from src.utils import load_checkpoint, denormalize


# ── Core inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (true_labels, predicted_labels, softmax_probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        with autocast():
            logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curves(y_true, y_probs, save_path: str):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — One vs. Rest", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_f1(report_dict: dict, save_path: str):
    classes = [c for c in report_dict if c not in ("accuracy", "macro avg", "weighted avg")]
    f1s = [report_dict[c]["f1-score"] for c in classes]
    short_names = [IDX_TO_CLASS.get(i, c) for i, c in enumerate(classes)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_names, f1s, color=plt.cm.RdYlGn(np.array(f1s)))
    ax.axhline(y=np.mean(f1s), color="navy", linestyle="--", linewidth=1.5, label=f"Mean F1={np.mean(f1s):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score on Test Set", fontsize=13, fontweight="bold")
    ax.legend()
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(config: dict):
    import os
    os.makedirs("results/figures", exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(device=device)
    load_checkpoint(model, path=config["checkpoint_path"], device=device)

    # ── Load test data ────────────────────────────────────────────────────────
    loaders = build_dataloaders(
        metadata_path=config["metadata_path"],
        image_dir=config["image_dir"],
        batch_size=config["batch_size"],
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    y_true, y_pred, y_probs = predict(model, loaders["test"], device)

    # ── Metrics ──────────────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    macro_auc = roc_auc_score(
        y_true, y_probs, multi_class="ovr", average="macro"
    )
    print(f"Macro AUC-ROC (OvR): {macro_auc:.4f}")

    metrics_out = {
        "accuracy":    report["accuracy"],
        "macro_f1":    report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_auc":   macro_auc,
        "per_class":   {c: report[c] for c in CLASS_NAMES},
    }

    with open("results/test_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("Saved: results/test_metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, "results/figures/confusion_matrix.png")
    plot_roc_curves(y_true, y_probs,      "results/figures/roc_curves.png")
    plot_per_class_f1(report,             "results/figures/per_class_f1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate skin lesion classifier")
    parser.add_argument("--metadata_path",   type=str, default="data/HAM10000_metadata.csv")
    parser.add_argument("--image_dir",       type=str, default="data/images")
    parser.add_argument("--checkpoint_path", type=str, default="results/best_model.pth")
    parser.add_argument("--batch_size",      type=int, default=64)
    args = parser.parse_args()
    evaluate(vars(args))
