from typing import Tuple

from torchvision import transforms
from torchvision.transforms import Compose

from cassava_model_comparison import config as cfg


# build_transforms
def build_transforms() -> Tuple[Compose, Compose, Compose]:

    train_transform = Compose([
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
    ])

    val_transform = Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
    ])

    test_transform = Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
    ])

    return train_transform, val_transform, test_transform