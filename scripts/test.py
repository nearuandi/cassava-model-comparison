import pandas as pd
import torch
import torch.nn as nn

from cassava_model_comparison import config as cfg
from cassava_model_comparison.datasets import build_dataloaders
from cassava_model_comparison.engine import evaluate_one_epoch
from cassava_model_comparison.engine import load_best_model


def main():
    data_dir = cfg.DATA_DIR
    train_dir = cfg.TRAIN_DIR
    runs_dir = cfg.RUNS_DIR

    train_csv = data_dir/ "train.csv"

    df = pd.read_csv(train_csv)
    df["image_path"] = train_dir / df["image_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"evaluate device: {device}")

    _, _, test_loader = build_dataloaders(df, device)

    loss_fn = nn.CrossEntropyLoss()

    model_list = [
        ("simple_cnn", runs_dir / "simple_cnn/best.pt"),
        ("mobilenet_v2", runs_dir / "mobilenet_v2/best.pt"),
        ("resnet18", runs_dir / "resnet18/best.pt"),
    ]


    for name, path in model_list:
        print(f"{name} 모델 테스트 시작")
        model = load_best_model(name, path, cfg.NUM_CLASSES).to(device)
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