import torch
import torch.nn as nn

from uc_ball_hyp_generator.layer.big_little_reduction import BigLittleReduction
from uc_ball_hyp_generator.layer.conv2dbn import Conv2dBn
from uc_ball_hyp_generator.layer.seblock import SEBlock


class NetworkV2(nn.Module):
    def __init__(self, input_height: int, input_width: int, num_classes: int = 2) -> None:
        super().__init__()

        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(3),
            Conv2dBn(3, 8, kernel_size=3, stride=1, padding=1, activation="relu"),
        )

        # Feature processing blocks
        self.feature_blocks = nn.ModuleList(
            [
                nn.Sequential(BigLittleReduction(8, 10, "relu"), nn.Dropout(0.2)),
                nn.Sequential(BigLittleReduction(15, 12, "relu"), SEBlock(18), nn.Dropout(0.1)),
                nn.Sequential(BigLittleReduction(18, 12, "relu"), SEBlock(18), nn.Dropout(0.1)),
                nn.Sequential(BigLittleReduction(18, 14, "relu"), SEBlock(21), nn.Dropout(0.1)),
            ]
        )

        # Modern classification head - only need to know final channel count
        final_channels = 126  # Output channels from last feature block
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, C, 1, 1) -> (B, C)
            nn.Linear(final_channels, 32),  # Slightly larger intermediate layer
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        for block in self.feature_blocks:
            x = block(x)

        # Adaptive pooling works for ANY spatial size
        x = self.classifier(x)  # (B, 21, 1, 1) -> (B, num_classes)

        return x


def create_network_v2(input_height: int, input_width: int) -> NetworkV2:
    return NetworkV2(input_height, input_width)
