# Third-party
import torch.nn as nn
from torchvision import models

# resnet18
def build_resnet18(
    num_classes: int,
    use_pretrained: bool = True
) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model