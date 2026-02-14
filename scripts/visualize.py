import torch

from cassava_model_comparison import config as cfg
from cassava_model_comparison.visualize import loss_curves, acc_curves, best_val_acc, multi_model_loss, multi_model_acc


runs_dir = cfg.RUNS_DIR

ckpt = torch.load(runs_dir / "simple_cnn/history.pt")
history = ckpt["history"]

loss_curves(history, "SimpleCNN")
acc_curves(history, "SimpleCNN")

best_val_acc(history, "SimpleCNN")

histories = {
    "simple_cnn": torch.load(runs_dir / "simple_cnn/history.pt")["history"],
    "mobilenet_v2": torch.load(runs_dir / "mobilenet_v2/history.pt")["history"],
    "resnet18": torch.load(runs_dir / "resnet18/history.pt")["history"],
}

multi_model_loss(histories, "Validation Loss")
multi_model_acc(histories, "Validation Accuracy (%)")
