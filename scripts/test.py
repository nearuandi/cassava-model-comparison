import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from omegaconf import DictConfig
import hydra

from cassava_model_comparison.datasets import build_dataloaders
from cassava_model_comparison.engine import evaluate_one_epoch
from cassava_model_comparison.engine import load_best_model

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.paths.data_dir)
    train_dir = Path(cfg.paths.train_dir)
    runs_dir = Path(cfg.paths.runs_dir)

    train_csv = data_dir/ "train.csv"

    df = pd.read_csv(train_csv)
    df["image_path"] = train_dir / df["image_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"test device: {device}")

    _, _, test_loader = build_dataloaders(df, cfg)

    loss_fn = nn.CrossEntropyLoss()

    model_list = [
        ("simple_cnn", runs_dir / "simple_cnn/best.pt"),
        ("mobilenet_v2", runs_dir / "mobilenet_v2/best.pt"),
        ("resnet18", runs_dir / "resnet18/best.pt"),
    ]


    for name, path in model_list:
        print(f"{name} 모델 테스트 시작")
        model = load_best_model(name, path, cfg.dataset.num_classes, device=device)
        model.to(device)
        model.eval()
        test_loss, test_acc = evaluate_one_epoch(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            device=device
        )
        print(f"{name:<12} | test loss: {test_loss:.4f} | test acc: {test_acc:.2f}%\n")


if __name__ == "__main__":
    main()