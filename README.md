---
title: PneumoNet - Pneumonia Detection
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# PneumoNet — Biomedical Image Classifier

Pneumonia detection from chest X-rays using two models: a custom CNN baseline and a fine-tuned ResNet50. Includes Grad-CAM, SHAP, and LIME explainability, plus a Gradio web interface.

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,216 train / 16 val / 624 test images.

---

## Results

| Model | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 74–75% | — | — | — | — |
| ResNet50 (fine-tuned) | 92–93% | 90.5% | 98.2% | 94.2% | ~97.5% |

---

## Project Structure

```
Biomedical_Image_Classifier/
├── config.yaml                  # all hyperparameters and paths
├── requirements.txt
├── src/
│   ├── data_loader.py           # transforms and DataLoaders
│   ├── models.py                # PneumoNetCNN, build_resnet50(), load_model()
│   ├── train.py                 # training loop, validation, weighted loss
│   ├── evaluate.py              # metrics, confusion matrix, ROC curve
│   ├── explain.py               # Grad-CAM, SHAP, LIME
│   └── app.py                   # Gradio web interface
├── notebooks/
│   ├── baseline_CNN.ipynb       # baseline CNN walkthrough
│   └── resnet50_finetune.ipynb  # ResNet50 two-stage fine-tuning
└── models/                      # place .pth weight files here (gitignored)
```

---

## Deployment

Deployed on **Hugging Face Spaces** — [live demo link here once deployed]

---

## Local Setup

**1. Install dependencies**

For running the app only:
```bash
pip install -r requirements.txt
```

For notebooks (training, evaluation, explainability):
```bash
pip install -r requirements_dev.txt
```

**2. Download the dataset**

Get it from Kaggle and place it at `chest_xray/` in the project root:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
```

**3. Download model weights** (optional — skip if training from scratch)

- CNN weights: https://drive.google.com/file/d/1Y39duosMWga5rfD8Jmiu4ezA0hqzLjxJ/view?usp=sharing
- ResNet50 weights: https://drive.google.com/file/d/1usp46btVZFXR4srEiHCJYQ2v9rn9_k3d/view?usp=sharing

Place both `.pth` files in the `models/` directory and rename them to match `config.yaml`:
```
models/pneumonet_cnn.pth
models/pneumonet_resnet50_finetuned.pth
```

---

## Usage

### Notebooks

Open either notebook from the `notebooks/` folder in Jupyter. Each notebook is self-contained and walks through data loading, training, evaluation, and explainability.

```bash
jupyter notebook notebooks/baseline_CNN.ipynb
jupyter notebook notebooks/resnet50_finetune.ipynb
```

### Gradio Web App (command line)

```bash
# ResNet50
python -m src.app --model resnet50 --weights models/pneumonet_resnet50_finetuned.pth

# Baseline CNN
python -m src.app --model cnn --weights models/pneumonet_cnn.pth
```

### Configuration

Edit `config.yaml` to change dataset path, batch size, learning rates, epochs, or model save paths — no need to touch any Python files.

---

## Models

### Baseline CNN (`src/models.py` → `PneumoNetCNN`)
3-layer convolutional network trained from scratch.
- 3× Conv → ReLU → MaxPool
- Fully connected: 256 neurons, Dropout(0.4), 2-class output
- Optimizer: Adam (lr=0.001), 10 epochs

### ResNet50 (`src/models.py` → `build_resnet50`)
Pretrained ImageNet backbone with two-stage training:
- **Stage 1** (12 epochs): freeze backbone, train classification head only
- **Stage 2** (5 epochs): unfreeze all layers, fine-tune end-to-end at lr=1e-5
- Class-weighted loss to handle the NORMAL/PNEUMONIA imbalance

---

## Explainability

All three methods are available via `src/explain.py` and demonstrated in both notebooks.

- **Grad-CAM** — highlights which regions of the X-ray drove the prediction
- **SHAP** — pixel-level attribution using GradientExplainer
- **LIME** — superpixel-based local explanation
