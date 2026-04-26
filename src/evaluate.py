import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)


def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_probs),
    }
    for k, v in metrics.items():
        print(f"{k.capitalize():12s}: {v:.4f}")
    return metrics, y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, classes=("NORMAL", "PNEUMONIA")):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_probs, roc_auc):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history["train_losses"], label="Train", marker='o')
    ax1.plot(history["val_losses"],   label="Val",   marker='o')
    ax1.set(title=f"Loss {title_suffix}", xlabel="Epoch", ylabel="Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot([a * 100 for a in history["train_accs"]], label="Train", marker='o')
    ax2.plot([a * 100 for a in history["val_accs"]],   label="Val",   marker='o')
    ax2.set(title=f"Accuracy {title_suffix}", xlabel="Epoch", ylabel="Accuracy (%)")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()
