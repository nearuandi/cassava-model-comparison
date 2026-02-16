from torchvision import transforms
from torchvision.transforms import Compose

from omegaconf import DictConfig


# build_transforms
def build_transforms(
        cfg: DictConfig
) -> tuple[Compose, Compose, Compose]:
    dataset = cfg.dataset

    train_transform = Compose([
        transforms.RandomResizedCrop(dataset.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(dataset.image_mean, dataset.image_std),
    ])

    val_transform = Compose([
        transforms.Resize((dataset.image_size, dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(dataset.image_mean, dataset.image_std),
    ])

    test_transform = Compose([
        transforms.Resize((dataset.image_size, dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(dataset.image_mean, dataset.image_std),
    ])

    return train_transform, val_transform, test_transform