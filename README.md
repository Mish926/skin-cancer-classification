# Skin Cancer Classification

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Multi-class skin lesion classification using ResNet50 transfer learning on the HAM10000 dataset. The pipeline is designed to handle severe class imbalance across 7 diagnostic categories and outputs clinically interpretable evaluation metrics.

## Table of Contents

- [Results](#results)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Design Decisions](#design-decisions)
- [References](#references)

---

## Results

Evaluated on a held-out test set of 1,121 images (stratified 70/15/15 split).

| Metric              | Value  |
|---------------------|--------|
| Test Accuracy       | 51.6%  |
| Macro F1            | 0.423  |
| Weighted F1         | 0.579  |
| Macro AUC-ROC (OvR) | 0.911  |

> Accuracy is suppressed by the 67% majority class (melanocytic nevi). Macro AUC-ROC is the primary metric — it measures discrimination ability uniformly across all 7 classes regardless of support.

**Per-class performance:**

| Class                | Precision | Recall | F1    | AUC   | Support |
|----------------------|-----------|--------|-------|-------|---------|
| Melanocytic nevi     | 0.992     | 0.480  | 0.647 | 0.895 | 811     |
| Melanoma             | 0.229     | 0.674  | 0.342 | 0.845 | 92      |
| Benign keratosis     | 0.370     | 0.468  | 0.413 | 0.843 | 109     |
| Basal cell carcinoma | 0.403     | 0.592  | 0.479 | 0.944 | 49      |
| Actinic keratoses    | 0.321     | 0.743  | 0.448 | 0.946 | 35      |
| Vascular lesions     | 0.359     | 1.000  | 0.528 | 0.996 | 14      |
| Dermatofibroma       | 0.055     | 0.636  | 0.101 | 0.910 | 11      |

**Training history**

![Training Curves](results/figures/training_curves.png)

**Confusion matrix**

![Confusion Matrix](results/figures/confusion_matrix.png)

**ROC curves — one vs. rest**

![ROC Curves](results/figures/roc_curves.png)

**Per-class F1**

![Per-Class F1](results/figures/per_class_f1.png)

---

## Dataset

HAM10000 — 10,015 dermoscopic images across 7 classes collected from two sites (Vienna and Queensland). Available on [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

| Class                | Abbreviation | Train count | Share  |
|----------------------|--------------|-------------|--------|
| Melanocytic nevi     | nv           | 4,657       | 67.0%  |
| Melanoma             | mel          | 634         | 9.1%   |
| Benign keratosis     | bkl          | 755         | 10.9%  |
| Basal cell carcinoma | bcc          | 340         | 4.9%   |
| Actinic keratoses    | akiec        | 241         | 3.5%   |
| Vascular lesions     | vasc         | 98          | 1.4%   |
| Dermatofibroma       | df           | 76          | 1.1%   |

---

## Architecture

ResNet50 pretrained on ImageNet with a replacement classifier head:

```
Input (224 x 224 x 3)
       |
ResNet50 Backbone
  conv1 + bn + relu + maxpool
  layer1  (64 -> 64,   3 blocks)   [frozen during warm-up]
  layer2  (64 -> 128,  4 blocks)   [frozen during warm-up]
  layer3  (128 -> 256, 6 blocks)
  layer4  (256 -> 512, 3 blocks)
  global average pool -> 2048-d
       |
Classifier Head
  Dropout(0.5)
  Linear(2048 -> 512) + ReLU + BatchNorm
  Dropout(0.3)
  Linear(512 -> 7)
```

**Two-phase training:**

| Phase       | Epochs | Backbone | LR (backbone) | LR (head) |
|-------------|--------|----------|---------------|-----------|
| Warm-up     | 1-5    | Frozen   | —             | 1e-3      |
| Fine-tuning | 6-30   | Unfrozen | 1e-5          | 1e-4      |

---

## Installation

```bash
git clone https://github.com/Mish926/skin-cancer-classification.git
cd skin-cancer-classification
pip install -r requirements.txt
```

**Dataset setup:**

Download HAM10000 from Kaggle and organize as follows:

```
dataset/
├── HAM10000_metadata.csv
└── images/           # contents of part_1 and part_2 merged
```

---

## Usage

**Train:**

```bash
python -m src.train \
  --metadata_path dataset/HAM10000_metadata.csv \
  --image_dir     dataset/images \
  --epochs        30 \
  --warmup_epochs 5 \
  --batch_size    16 \
  --patience      7 \
  --checkpoint_path results/best_model.pth \
  --num_workers   0
```

**Evaluate:**

```bash
python -m src.evaluate \
  --metadata_path   dataset/HAM10000_metadata.csv \
  --image_dir       dataset/images \
  --checkpoint_path results/best_model.pth
```

**MLflow dashboard:**

```bash
mlflow ui   # http://localhost:5000
```

---

## Repository Structure

```
skin-cancer-classification/
├── src/
│   ├── dataset.py       # Data loading, augmentation, weighted sampling
│   ├── model.py         # ResNet50 with custom head
│   ├── train.py         # Training loop, focal loss, MLflow logging
│   ├── evaluate.py      # Test metrics, confusion matrix, ROC curves
│   └── utils.py         # Checkpointing, visualization
├── notebook/
│   └── EDA_and_Results.ipynb
├── results/
│   ├── test_metrics.json
│   └── figures/
├── requirements.txt
└── README.md
```

---

## Design Decisions

**WeightedRandomSampler over class-weighted loss alone**
The dataset is heavily skewed (nv: 67%, df: 1.1%). Oversampling minority classes at batch construction ensures each batch contains balanced representation, which is more effective than post-hoc loss weighting for extreme imbalance.

**Focal Loss (gamma=2)**
Standard cross-entropy treats all samples equally. Focal loss downweights easy, confidently classified examples and redirects gradient updates toward hard minority cases — important when the majority class is learned quickly.

**Frozen backbone warm-up**
Initializing the classifier head randomly means early gradients are large and unstable. Freezing the backbone for the first 5 epochs prevents these gradients from corrupting the pretrained ImageNet features before the head has stabilized.

**Discriminative learning rates**
The backbone layers already encode general visual features and require only fine adjustment (LR: 1e-5). The head is trained from scratch and needs a larger learning rate (LR: 1e-4) to converge in the available epochs.

**Augmentation strategy**
`RandomResizedCrop` and `ColorJitter` simulate dermoscope variation across devices and lighting conditions. Horizontal and vertical flips are both valid — skin lesions have no canonical orientation.

---

## References

Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161. https://doi.org/10.1038/sdata.2018.161

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.

Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. *ICCV 2017*.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
