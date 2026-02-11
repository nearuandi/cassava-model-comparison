# Third-party
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local
from cassava_model_comparison import config as cfg
from cassava_model_comparison.datasets import build_dataloaders
from cassava_model_comparison.engine import run_training
from cassava_model_comparison.models import build_model


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

    simple_cnn_model = build_model("simple_cnn", cfg.NUM_CLASSES).to(device)
    simple_cnn_optimizer = optim.Adam(simple_cnn_model.parameters(), lr=cfg.LEARNING_RATE)
    simple_cnn_scaler = GradScaler(enabled=(device.type == "cuda"))
    simple_cnn_scheduler = ReduceLROnPlateau(
        optimizer = simple_cnn_optimizer,
        mode = cfg.LR_MODE,
        factor = cfg.LR_FACTOR,
        patience = cfg.LR_PATIENCE,
    )

    simple_cnn_dir = runs_dir / "simple_cnn"
    run_training(
        run_name="simple_cnn",
        run_dir=simple_cnn_dir,
        epochs=cfg.EPOCHS,
        model=simple_cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        optimizer=simple_cnn_optimizer,
        scaler=simple_cnn_scaler,
        scheduler=simple_cnn_scheduler
    )


    mobilenet_v2_model = build_model("mobilenet_v2", cfg.NUM_CLASSES).to(device)
    mobilenet_v2_optimizer = optim.Adam(mobilenet_v2_model.parameters(), lr=cfg.LEARNING_RATE)
    mobilenet_v2_scaler = GradScaler(enabled=(device.type == "cuda"))
    mobilenet_v2_scheduler = ReduceLROnPlateau(
        optimizer = mobilenet_v2_optimizer,
        mode = cfg.LR_MODE,
        factor = cfg.LR_FACTOR,
        patience = cfg.LR_PATIENCE,
    )

    mobilenet_v2_dir = runs_dir / "mobilenet_v2"
    run_training(
        run_name="mobilenet_v2",
        run_dir=mobilenet_v2_dir,
        epochs=cfg.EPOCHS,
        model=mobilenet_v2_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        optimizer=mobilenet_v2_optimizer,
        scaler=mobilenet_v2_scaler,
        scheduler=mobilenet_v2_scheduler
    )

    resnet18_model = build_model("resnet18", cfg.NUM_CLASSES).to(device)
    resnet18_optimizer = optim.Adam(resnet18_model.parameters(), lr=cfg.LEARNING_RATE)
    resnet18_scaler = GradScaler(enabled=(device.type == "cuda"))
    resnet18_scheduler = ReduceLROnPlateau(
        optimizer = resnet18_optimizer,
        mode = cfg.LR_MODE,
        factor = cfg.LR_FACTOR,
        patience = cfg.LR_PATIENCE,
    )

    resnet18_dir = runs_dir / "resnet18"
    run_training(
        run_name="resnet18",
        run_dir=resnet18_dir,
        epochs=cfg.EPOCHS,
        model=resnet18_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        optimizer=resnet18_optimizer,
        scaler=resnet18_scaler,
        scheduler=resnet18_scheduler
    )

if __name__ == "__main__":
    main()