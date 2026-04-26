import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, y_true, y_pred = 0.0, [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    return running_loss / len(loader), accuracy_score(y_true, y_pred)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return val_loss / len(loader), accuracy_score(y_true, y_pred)


def get_weighted_criterion(train_data, device):
    counts = np.bincount(train_data.targets)
    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def train_loop(model, train_loader, val_loader, criterion, optimizer,
               num_epochs, device, save_path=None, scheduler=None):
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if scheduler:
            scheduler.step()

    return history
