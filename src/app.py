"""
Launch: python -m src.app --model resnet50 --weights models/pneumonet_resnet50_finetuned.pth
        python -m src.app --model cnn      --weights models/pneumonet_cnn.pth
"""
import argparse
import torch
import gradio as gr
from torchvision import transforms
from .models import PneumoNetCNN, build_resnet50, load_model

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn(weights_path):
    return load_model(PneumoNetCNN(), weights_path, device)


def load_resnet(weights_path):
    return load_model(build_resnet50(freeze_backbone=False), weights_path, device)


def make_predict_fn(model):
    def predict(img):
        tensor = _transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        return {"NORMAL": probs[0].item(), "PNEUMONIA": probs[1].item()}
    return predict


def launch_app(model, title="PneumoNet - Pneumonia Detection"):
    gr.Interface(
        fn=make_predict_fn(model),
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=2),
        title=title,
        description="Upload a chest X-ray image to predict whether it's Pneumonia or Normal.",
    ).launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet50"], default="resnet50")
    parser.add_argument("--weights", required=True, help="Path to .pth weights file")
    args = parser.parse_args()

    m = load_cnn(args.weights) if args.model == "cnn" else load_resnet(args.weights)
    title = f"PneumoNet {'CNN' if args.model == 'cnn' else 'ResNet50'} - Pneumonia Detection"
    launch_app(m, title)
