# Standard library
from typing import Tuple

# Third-party
from torchvision import transforms
from torchvision.transforms import Compose

# Local
from cassava_model_comparison import config as cfg


# build_transforms
def build_transforms() -> Tuple[Compose, Compose, Compose]:

    train_tf = Compose([
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
    ])

    val_tf = Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
    ])

    test_tf = Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
    ])

    return train_tf, val_tf, test_tf