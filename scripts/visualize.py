# Third-party
import torch

# Local
from cassava_model_comparison import config as cfg
from cassava_model_comparison.visualize.plots import *


runs_dir = cfg.RUNS_DIR

ckpt = torch.load(runs_dir / "simple_cnn/history.pt", weights_only=False)  # history 저장 파일
history = ckpt["history"]

plot_loss_curves(history, title="simple_cnn Loss")
plot_acc_curves(history, title="simple_cnn Acc")

plot_best_val_acc(history, title="simple_cnn best_val_acc")

histories = {
    "simple_cnn": torch.load(runs_dir / "simple_cnn/history.pt")["history"],
    "mobilenet_v2": torch.load(runs_dir / "mobilenet_v2/history.pt")["history"],
    "resnet18": torch.load(runs_dir / "resnet18/history.pt")["history"],
}

plot_multi_model_loss(histories, "Loss Comparison")
plot_multi_model_acc(histories, "Accuracy Comparison")
