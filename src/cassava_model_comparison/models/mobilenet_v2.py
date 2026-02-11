# Third-party
import torch.nn as nn
from torchvision import models

# mobilenet_v2
def build_mobilenet_v2(
    num_classes: int,
    use_pretrained: bool = True
) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
    model = models.mobilenet_v2(weights=weights)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model