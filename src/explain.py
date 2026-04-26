import copy
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


def _unnormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(_STD * img + _MEAN, 0, 1)


def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def generate_gradcam(model, img_tensor, target_layer):
    model.eval()
    gradients, activations = [], []

    h_f = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    h_b = target_layer.register_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))

    outputs = model(img_tensor)
    pred_class = outputs.argmax(dim=1)
    model.zero_grad()
    outputs[0, pred_class].backward()

    h_f.remove()
    h_b.remove()

    grads = gradients[0].cpu().numpy()[0]
    acts  = activations[0].cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.maximum(np.sum([w * a for w, a in zip(weights, acts)], axis=0), 0)
    cam /= cam.max() if cam.max() != 0 else 1
    return cam, pred_class.item()


def show_gradcam(model, sample_img, target_layer, device):
    cam, pred = generate_gradcam(model, sample_img.unsqueeze(0).to(device), target_layer)
    img_disp    = np.uint8(_unnormalize(sample_img) * 255)
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.4, img_disp, 0.6, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay[..., ::-1])
    plt.title(f"Grad-CAM (Pred: {'PNEUMONIA' if pred == 1 else 'NORMAL'})")
    plt.axis('off')
    plt.show()


def show_shap(model, test_imgs, device, n_samples=4):
    model_copy = copy.deepcopy(model).to(device).eval()
    samples    = test_imgs[:n_samples].to(device)
    background = test_imgs[:1].to(device)

    e  = shap.GradientExplainer(model_copy, background)
    sv = e.shap_values(samples)  # (N, C, H, W, K)

    N, C, H, W, K = sv.shape
    shap_list = [np.transpose(sv[..., k], (0, 2, 3, 1)) for k in range(K)]
    imgs_np   = samples.detach().cpu().permute(0, 2, 3, 1).numpy()
    shap.image_plot(shap_list, imgs_np)
    plt.show()


def show_lime(model, img_tensor, device):
    def predict_fn(images):
        t = torch.tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32).to(device)
        with torch.no_grad():
            return torch.softmax(model(t), dim=1).cpu().numpy()

    img_np = img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    exp = lime_image.LimeImageExplainer().explain_instance(
        img_np, predict_fn, top_labels=2, hide_color=0, num_samples=1000
    )
    temp, mask = exp.get_image_and_mask(
        exp.top_labels[0], positive_only=False, num_features=10, hide_rest=False
    )
    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME (Pred: {'PNEUMONIA' if exp.top_labels[0] == 1 else 'NORMAL'})")
    plt.axis('off')
    plt.show()
