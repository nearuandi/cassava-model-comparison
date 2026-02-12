# Standard library
import time
from pathlib import Path

# Third-party
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Local
from cassava_model_comparison import config as cfg
from cassava_model_comparison.datasets import build_dataloaders
from cassava_model_comparison.models import build_model
from cassava_model_comparison.engine import train_one_epoch, evaluate_one_epoch, save_best, save_history

def fit(
        run_name: str,
        run_dir: Path,
        epochs: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
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
        model.train()
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=scaler
        )
        model.eval()
        val_loss, val_acc = evaluate_one_epoch(
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
    print(f"{run_name} 모델 훈련 완료, train_time: {train_time:.1f}초, best_val_acc: {best_val_acc:.2f}\n")


def main():
    data_dir = cfg.DATA_DIR
    train_dir = cfg.TRAIN_DIR
    runs_dir = cfg.RUNS_DIR

    train_csv = data_dir / "train.csv"

    df = pd.read_csv(train_csv)
    df["image_path"] = train_dir / df["image_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train device: {device}")

    train_loader, val_loader, _ = build_dataloaders(df, device)

    loss_fn = nn.CrossEntropyLoss()

    model_names = ["simple_cnn", "mobilenet_v2", "resnet18"]

    for model_name in model_names:
        model = build_model(model_name, cfg.NUM_CLASSES).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        scaler = GradScaler(enabled=(device.type == "cuda"))
        scheduler = ReduceLROnPlateau(optimizer, mode=cfg.LR_MODE, factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE)

        fit(
            run_name=model_name,
            run_dir=runs_dir / model_name,
            epochs=cfg.EPOCHS,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
        )

if __name__ == "__main__":
    main()