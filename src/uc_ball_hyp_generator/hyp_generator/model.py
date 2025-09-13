import torch
import torch.nn as nn
import torch.nn.functional as F

from uc_ball_hyp_generator.layer.big_little_reduction import BigLittleReduction
from uc_ball_hyp_generator.layer.conv2dbn import Conv2dBn
from uc_ball_hyp_generator.layer.seblock import SEBlock


class NetworkV2(nn.Module):
    def __init__(self, input_height: int, input_width: int, num_classes: int = 3) -> None:
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
        self.coord_xy_classifier = nn.Sequential(
            nn.Flatten(),  # (B, C, 1, 1) -> (B, C)
            nn.Linear(final_channels, 32),  # Slightly larger intermediate layer
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        for block in self.feature_blocks:
            x = block(x)

        # Adaptive pooling works for ANY spatial size
        x = self.coord_xy_classifier(x)  # (B, 21, 1, 1) -> (B, num_classes)

        return x


class NetworkV2Hypercolumn(nn.Module):
    def __init__(self, input_height: int, input_width: int, num_classes: int = 3) -> None:
        super().__init__()

        # Feature extraction backbone with doubled filters
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(3),
            # Original filters: 8 -> New filters: 16
            Conv2dBn(3, 16, kernel_size=3, stride=1, padding=1, activation="relu"),
        )

        # Feature processing blocks with doubled filters
        self.feature_blocks = nn.ModuleList(
            [
                # Block 0: in=16, filter_count=20 -> out=(20/2)+20=30
                nn.Sequential(BigLittleReduction(16, 20, "relu"), nn.Dropout(0.2)),
                # Block 1: in=30, filter_count=24 -> out=(24/2)+24=36
                nn.Sequential(BigLittleReduction(30, 24, "relu"), SEBlock(36), nn.Dropout(0.1)),
                # Block 2: in=36, filter_count=24 -> out=(24/2)+24=36
                nn.Sequential(BigLittleReduction(36, 24, "relu"), SEBlock(36), nn.Dropout(0.1)),
                # Block 3: in=36, filter_count=28 -> out=(28/2)+28=42
                nn.Sequential(BigLittleReduction(36, 28, "relu"), SEBlock(42), nn.Dropout(0.1)),
            ]
        )

        # Hypercolumn classifier with updated input size
        # New total channels: 16 + 30 + 36 + 36 + 42 = 160
        hypercolumn_channels = 160
        self.hypercolumn_classifier = nn.Sequential(
            nn.Linear(hypercolumn_channels, 64),  # Increased intermediate layer size
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = []

        # 1. Pass through backbone and capture its output
        x = self.backbone(x)
        feature_maps.append(x)

        # 2. Pass through each feature block and capture their outputs
        for block in self.feature_blocks:
            x = block(x)
            feature_maps.append(x)

        # 3. Pool each captured feature map to a vector and flatten
        pooled_vectors = [torch.flatten(F.adaptive_avg_pool2d(fm, (1, 1)), 1) for fm in feature_maps]

        # 4. Concatenate all vectors to form the hypercolumn
        hypercolumn_vector = torch.cat(pooled_vectors, dim=1)

        # 5. Pass the rich hypercolumn vector to the new classifier
        out = self.hypercolumn_classifier(hypercolumn_vector)

        return out


def get_ball_hyp_model(input_height: int, input_width: int) -> nn.Module:
    return NetworkV2Hypercolumn(input_height=input_height, input_width=input_width, num_classes=3)
