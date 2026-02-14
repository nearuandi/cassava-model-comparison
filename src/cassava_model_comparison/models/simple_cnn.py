# Third-party
import torch.nn as nn
from torch import Tensor

# SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block1: (B,32,H/2,W/2)
            self.conv_block(3, 32),

            # Block2: (B,64,H/4,W/4)
            self.conv_block(32, 64),

            # Block3: (B,128,H/8,W/8)
            self.conv_block(64, 128),

            # Block4: (B,256,H/16,W/16)
            self.conv_block(128, 256),

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

    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x