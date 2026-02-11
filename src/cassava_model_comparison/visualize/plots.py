# Standard library
from typing import Dict, List, Tuple

# Third-party
import matplotlib.pyplot as plt


History = Dict[str, List[float]]


def _get_epochs(history: History) -> List[int]:
    n = len(history.get("val_loss", []))
    return list(range(1, n + 1))


def plot_loss_curves(
        history: History,
        title: str = "Loss Curves"
) -> None:
    epochs = _get_epochs(history)
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acc_curves(
        history: History,
        title: str = "Accuracy Curves (%)"
) -> None:
    epochs = _get_epochs(history)
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_best_val_acc(
        history: History,
        title: str
) -> None:
    epochs = _get_epochs(history)
    val_acc = history["val_acc"]

    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    best_epoch = best_idx + 1
    best_val = val_acc[best_idx]

    plt.figure()
    plt.plot(epochs, val_acc, label="val_acc")
    plt.scatter([best_epoch], [best_val])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multi_model_loss(
        histories: dict,
        title: str
) -> None:
    plt.figure()

    for name, history in histories.items():
        epochs = _get_epochs(history)
        plt.plot(epochs, history["val_loss"], label=f"{name}_val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multi_model_acc(
        histories: dict,
        title: str
) -> None:
    plt.figure()

    for name, history in histories.items():
        epochs = _get_epochs(history)
        plt.plot(epochs, history["val_acc"], label=f"{name}_val")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()