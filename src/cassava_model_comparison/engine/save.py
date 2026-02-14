from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cassava_model_comparison.models import build_model


def save_best(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: GradScaler,
    best_val_acc: float
) -> None:
    best = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": best_val_acc
    }
    torch.save(best, run_dir / "best.pt")


def save_history(
    run_dir: Path,
    history: Dict[str, list],
    train_time: float,
    best_val_acc: float
) -> None:
    history_data = {
        "history": history,
        "train_time": train_time,
        "best_val_acc": best_val_acc
    }

    torch.save(history_data, run_dir / "history.pt")


def load_best_model(
    model_name: str,
    ckpt_path: str,
    num_classes: int
) -> nn.Module:
    ckpt = torch.load(ckpt_path)
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    return model