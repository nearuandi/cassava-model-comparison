# Third-party
import torch.nn as nn

# Local
from .simple_cnn import SimpleCNN
from .mobilenet_v2 import build_mobilenet_v2
from .resnet18 import build_resnet18


# build_model
def build_model(
        model_name: str,
        num_classes: int,
        use_pretrained: bool = True
) -> nn.Module:
    model_name = model_name.lower()

    if model_name in {"simple_cnn", "simplecnn"}:
        return SimpleCNN(num_classes=num_classes)

    if model_name in {"mobilenet_v2", "mobilenetv2"}:
        return build_mobilenet_v2(num_classes=num_classes, use_pretrained=use_pretrained)

    if model_name in {"resnet18"}:
        return build_resnet18(num_classes=num_classes, use_pretrained=use_pretrained)

    raise ValueError(f"simple_cnn, mobilenet_v2, resnet18 모델만")