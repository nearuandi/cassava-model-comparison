# Standard library
from typing import Tuple

# Third-party
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local
from cassava_model_comparison import config as cfg
from cassava_model_comparison.datasets import CassavaDataset, build_transforms, split_train_val_test


def build_dataloaders(
        df: pd.DataFrame,
        device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_df, val_df, test_df = split_train_val_test(df, random_seed=cfg.SEED)

    train_transform, val_transform, test_transform = build_transforms()

    train_dataset = CassavaDataset(train_df, transform=train_transform)
    val_dataset = CassavaDataset(val_df, transform=val_transform)
    test_dataset = CassavaDataset(test_df, transform=test_transform)

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader