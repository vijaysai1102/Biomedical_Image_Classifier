import os
import sys
import torch
import gradio as gr
import gdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import build_resnet50, load_model
from src.app import make_predict_fn

MODEL_PATH = "models/pneumonet_resnet50_finetuned.pth"
GDRIVE_ID  = "1usp46btVZFXR4srEiHCJYQ2v9rn9_k3d"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model weights...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_model(build_resnet50(freeze_backbone=False), MODEL_PATH, device)

demo = gr.Interface(
    fn=make_predict_fn(model),
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="PneumoNet - Pneumonia Detection",
    description="Upload a chest X-ray image to get a prediction: Normal or Pneumonia.",
)

demo.launch()
