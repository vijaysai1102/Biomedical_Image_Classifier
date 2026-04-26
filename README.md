---
title: PneumoNet - Pneumonia Detection
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.23.3"
python_version: "3.10"
app_file: app.py
pinned: false
---

# PneumoNet — Biomedical Image Classifier

Pneumonia detection from chest X-rays using deep learning. Two models are implemented — a custom CNN baseline and a fine-tuned ResNet50 — with Grad-CAM, SHAP, and LIME explainability.

**Live Demo:** [huggingface.co/spaces/Vijaysai16/pneumonet](https://huggingface.co/spaces/Vijaysai16/pneumonet)

**Dataset:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
5,216 training / 16 validation / 624 test images across two classes: NORMAL and PNEUMONIA.

---

## Results

| Model | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN (from scratch) | 74–75% | — | — | — | — |
| ResNet50 (fine-tuned) | 92–93% | 90.5% | 98.2% | 94.2% | ~97.5% |

---

## Project Structure

```
Biomedical_Image_Classifier/
├── app.py                        # Hugging Face Spaces entry point
├── requirements.txt              # inference dependencies (app only)
├── requirements_dev.txt          # full dependencies (notebooks + app)
├── config.yaml                   # all hyperparameters and paths
│
├── src/
│   ├── models.py                 # PneumoNetCNN and build_resnet50()
│   ├── data_loader.py            # transforms and DataLoaders
│   ├── train.py                  # training loop and validation
│   ├── evaluate.py               # metrics, confusion matrix, ROC curve
│   ├── explain.py                # Grad-CAM, SHAP, LIME
│   └── app.py                    # Gradio inference logic
│
├── notebooks/
│   ├── baseline_CNN.ipynb        # baseline CNN: train, evaluate, explain
│   └── resnet50_finetune.ipynb   # ResNet50: two-stage fine-tuning, explain
│
└── models/                       # model weights — gitignored, download below
    ├── pneumonet_cnn.pth
    └── pneumonet_resnet50_finetuned.pth
```

---

## Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/vijaysai1102/Biomedical_Image_Classifier.git
cd Biomedical_Image_Classifier
```

**2. Install dependencies**

To run the web app only:
```bash
pip install -r requirements.txt
```

To run notebooks (training, evaluation, explainability):
```bash
pip install -r requirements_dev.txt
```

**3. Download the dataset**

Download from Kaggle and place it in the project root:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
```

**4. Download model weights**

- CNN: https://drive.google.com/file/d/1Y39duosMWga5rfD8Jmiu4ezA0hqzLjxJ/view?usp=sharing
- ResNet50: https://drive.google.com/file/d/1usp46btVZFXR4srEiHCJYQ2v9rn9_k3d/view?usp=sharing

Place them in `models/`:
```
models/pneumonet_cnn.pth
models/pneumonet_resnet50_finetuned.pth
```

---

## Usage

### Run the web app locally
```bash
python app.py
```
Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Run notebooks
```bash
jupyter notebook notebooks/baseline_CNN.ipynb
jupyter notebook notebooks/resnet50_finetune.ipynb
```

### Configuration
All hyperparameters (learning rate, batch size, epochs, paths) are in `config.yaml`. No Python files need to be edited to change training settings.

---

## Models

### Baseline CNN
Defined in `src/models.py` as `PneumoNetCNN`. A 3-layer convolutional network trained from scratch.

- Architecture: 3× (Conv2d → ReLU → MaxPool), then Flatten → Linear(256) → ReLU → Dropout(0.4) → Linear(2)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001), 10 epochs
- Limitation: overfits — 97% train accuracy but only 74% test accuracy

### ResNet50 (Fine-tuned)
Defined in `src/models.py` as `build_resnet50()`. Pretrained on ImageNet, adapted for binary classification with two-stage training.

- **Stage 1** (12 epochs): backbone frozen, only the classification head trained (lr=1e-4)
- **Stage 2** (5 epochs): all layers unfrozen, end-to-end fine-tuning (lr=1e-5)
- Loss: class-weighted CrossEntropyLoss to handle NORMAL/PNEUMONIA imbalance
- Scheduler: StepLR (step=3, gamma=0.1)

---

## Explainability

All three methods are in `src/explain.py` and demonstrated in both notebooks.

| Method | What it shows |
|---|---|
| **Grad-CAM** | Heatmap of which lung regions influenced the prediction |
| **SHAP** | Pixel-level attribution using GradientExplainer |
| **LIME** | Superpixel-based local explanation of a single prediction |
