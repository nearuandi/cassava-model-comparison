import pandas as pd
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from cassava_model_comparison.datasets import CassavaDataset, build_transforms, split_train_val_test


def build_dataloaders(
        df: pd.DataFrame,
        cfg: DictConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = cfg.train

    train_df, val_df, test_df = split_train_val_test(df, random_seed=train.seed)

    train_transform, val_transform, test_transform = build_transforms(cfg)

    train_dataset = CassavaDataset(train_df, transform=train_transform)
    val_dataset = CassavaDataset(val_df, transform=val_transform)
    test_dataset = CassavaDataset(test_df, transform=test_transform)


    train_loader = DataLoader(
        train_dataset,
        batch_size=train.batch_size,
        shuffle=True,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train.batch_size,
        shuffle=False,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train.batch_size,
        shuffle=False,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=False
    )
    return train_loader, val_loader, test_loader