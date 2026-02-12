# Standard library
from typing import Tuple

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

# Local
from cassava_model_comparison.models import build_model


@torch.no_grad()
def evaluate_one_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device
) -> Tuple[float, float]:

    running_loss = 0.0
    num_correct = 0
    num_samples = 0

    use_amp = (device.type == "cuda")

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        batch_size = labels.size(dim=0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        num_correct += preds.eq(labels).sum().item()
        num_samples += batch_size

    val_loss = running_loss / max(num_samples, 1)
    val_acc = 100.0 * num_correct / max(num_samples, 1)

    return val_loss, val_acc


@torch.no_grad()
def collect_predictions(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)



