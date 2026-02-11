# Standard library
import time
from pathlib import Path

# Third-party
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Local
from .train import train_one_epoch
from .eval import validate_one_epoch
from .save import save_best, save_history


def run_training(
        run_name: str,
        epochs: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        run_dir: Path,
        optimizer: Optimizer,
        scaler: GradScaler,
        scheduler: ReduceLROnPlateau,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    best_val_acc = 0.0
    print(f"{run_name} 모델 훈련 시작")
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=scaler
        )
        val_loss, val_acc = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )
        scheduler.step(val_loss)

        print(
            f"\n[Epoch {epoch + 1:02d}/{epochs}] "
            f"{run_name} | "
            f"Train: Loss {train_loss:.4f}, Acc {train_acc:6.2f}% | "
            f"Val: Loss {val_loss:.4f}, Acc {val_acc:6.2f}%"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best(
                run_dir=run_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_val_acc=best_val_acc
            )

    train_time = time.time() - start_time
    save_history(
        run_dir=run_dir,
        history=history,
        train_time=train_time,
        best_val_acc=best_val_acc,
    )
    print(f"{run_name} 모델 훈련 완료, best_val_acc: {best_val_acc:.2f}\n")
