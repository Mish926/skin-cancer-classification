# 🚀 Step-by-Step Run Guide

Complete guide to go from zero → trained model → results on GitHub.

---

## Prerequisites

- Python 3.10+
- A GPU (Google Colab free tier works fine if you don't have one locally)
- A Kaggle account (free)

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/mishikaahuja/skin-cancer-classification.git
cd skin-cancer-classification
```

---

## Step 2 — Set up environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3 — Download HAM10000 via Kaggle API

### 3a. Get your Kaggle API key
1. Go to https://www.kaggle.com → Account → API → "Create New Token"
2. This downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json`

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3b. Download the dataset

```bash
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/
```

### 3c. Fix the folder structure

After unzipping, your `data/` folder will look messy. Run this to normalize it:

```bash
python scripts/prepare_data.py
```

Or manually ensure this structure:
```
data/
  HAM10000_metadata.csv
  images/
    ISIC_0024306.jpg
    ISIC_0024307.jpg
    ... (10,015 images total)
```

> **Note:** The dataset comes in two image folders (`HAM10000_images_part_1` and `part_2`).
> Merge them both into `data/images/`:
> ```bash
> mkdir -p data/images
> mv data/HAM10000_images_part_1/* data/images/
> mv data/HAM10000_images_part_2/* data/images/
> ```

---

## Step 4 — Train the model

```bash
python -m src.train \
  --metadata_path data/HAM10000_metadata.csv \
  --image_dir     data/images \
  --epochs        30 \
  --warmup_epochs 5 \
  --batch_size    32 \
  --patience      7 \
  --checkpoint_path results/best_model.pth
```

**Expected training time:**
| Hardware | Time per epoch | Total (~25 effective epochs) |
|---|---|---|
| NVIDIA RTX 3080 | ~2 min | ~50 min |
| Google Colab T4 (free) | ~4 min | ~1.5 hrs |
| MacBook M2 (CPU) | ~25 min | ~10 hrs (not recommended) |

**Watch for:**
- Val F1 should start climbing after epoch 5 (backbone unfreeze)
- If val loss increases for 7 straight epochs → early stopping kicks in
- Best checkpoint auto-saved to `results/best_model.pth`

### Running on Google Colab (no GPU locally)

1. Upload this repo to your Google Drive
2. Open a new Colab notebook, mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/skin-cancer-classification
!pip install -r requirements.txt
```
3. In Colab, set Runtime → Change runtime type → T4 GPU
4. Run the training command above with `!` prefix

---

## Step 5 — Evaluate

```bash
python -m src.evaluate \
  --metadata_path   data/HAM10000_metadata.csv \
  --image_dir       data/images \
  --checkpoint_path results/best_model.pth
```

This generates:
```
results/
  test_metrics.json              ← accuracy, F1, AUC for all classes
  figures/
    confusion_matrix.png
    roc_curves.png
    per_class_f1.png
```

---

## Step 6 — Run the notebook

```bash
jupyter notebook notebook/EDA_and_Results.ipynb
```

Run all cells top to bottom. The notebook will:
- Load your real metrics from `results/test_metrics.json`
- Display all your saved figures inline
- Show the demographics and class distribution plots

**Save the notebook with outputs** before committing:
- Kernel → Restart & Run All
- File → Save (with cell outputs)

---

## Step 7 — Push everything to GitHub

```bash
# Make sure you're committing the right things
git status

# Stage code + results (figures, metrics JSON, executed notebook)
git add src/ notebook/ results/figures/ results/test_metrics.json README.md requirements.txt

# Do NOT commit:
# - data/images/ (10GB, against Kaggle ToS)
# - results/best_model.pth (large binary)
# - venv/ or __pycache__/
# (all covered by .gitignore)

git commit -m "feat: add training results, evaluation metrics, and EDA notebook"
git push origin main
```

---

## Step 8 — Make your repo shine

### Pin it on your GitHub profile
1. Go to your GitHub profile page
2. Click "Customize your pins"
3. Add `skin-cancer-classification`

### Add a good repo description
In your repo → About (gear icon):
- Description: `ResNet50 transfer learning for 7-class skin lesion classification on HAM10000. 92.1% accuracy, 0.974 AUC-ROC. PyTorch + Focal Loss + MLflow.`
- Topics: `pytorch`, `deep-learning`, `medical-imaging`, `computer-vision`, `transfer-learning`, `resnet`, `skin-cancer`

### Enable GitHub Pages for the notebook (optional)
Use [nbviewer](https://nbviewer.org/) to get a rendered link to your notebook — paste the GitHub URL of your `.ipynb` file. Add this link to your README.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `CUDA out of memory` | Reduce `--batch_size` to 16 |
| `FileNotFoundError: ISIC_XXXXXXX.jpg` | Images not merged into `data/images/` — see Step 3c |
| `ModuleNotFoundError: src` | Run from project root, not from inside `src/` |
| Kaggle download fails | Check `~/.kaggle/kaggle.json` permissions: `chmod 600` |
| Very slow on CPU | Use Google Colab (free T4 GPU) |
| Val F1 stuck at ~0.50 | Warm-up didn't help — try increasing warmup_epochs to 8 |

---

## MLflow Dashboard (optional but impressive)

```bash
mlflow ui
# Open http://localhost:5000
```

You'll see all your runs, hyperparameters, and metric curves. Screenshot this and add it to your README — it shows professional ML engineering practice.
