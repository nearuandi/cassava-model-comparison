# Third-party
import torch.nn as nn
from torch import Tensor

# SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block1: (B,32,H/2,W/2)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block2: (B,64,H/4,W/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block3: (B,128,H/8,W/8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block4: (B,256,H/16,W/16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # GAP: (B,256,1,1)
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            # (B,256)
            nn.Flatten(1),

            # (B,512)
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # (B,num_classes)
            nn.Linear(512, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x