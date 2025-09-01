import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(act: str) -> nn.Module:
    if act == "relu":
        return nn.ReLU()
    elif act == "leakyrelu":
        return nn.LeakyReLU()
    elif act == "relu6":
        return nn.ReLU6()
    elif act == "elu":
        return nn.ELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()

    raise NotImplementedError()


class Conv2dBn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_filter: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, nb_filter, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nb_filter, affine=False)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BigLittleReduction(nn.Module):
    def __init__(self, in_channels: int, filter_count: int, activation: str = "relu") -> None:
        super().__init__()
        self.conv1_3x3 = Conv2dBn(
            in_channels, filter_count // 2, kernel_size=3, stride=2, padding=1, activation=activation
        )
        self.conv2_3x3 = Conv2dBn(in_channels, filter_count, kernel_size=3, stride=1, padding=1, activation=activation)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_out = self.conv1_3x3(x)
        conv2_out = self.conv2_3x3(x)
        conv2_out = self.avg_pool(conv2_out)
        return torch.cat([conv1_out, conv2_out], dim=1)


class SEBlock(nn.Module):
    def __init__(self, input_channels: int, r: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // r)
        self.fc2 = nn.Linear(input_channels // r, input_channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.pool(input_tensor)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return input_tensor * x


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

        # Modern adaptive pooling - no manual size calculation needed!
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Always outputs (B, C, 1, 1)

        # Modern classification head - only need to know final channel count
        final_channels = 21  # Output channels from last feature block
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, C, 1, 1) -> (B, C)
            nn.Linear(final_channels, 64),  # Slightly larger intermediate layer
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout for better generalization
            nn.Linear(64, 32),
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
        x = self.global_pool(x)  # (B, 21, H, W) -> (B, 21, 1, 1)
        x = self.classifier(x)  # (B, 21, 1, 1) -> (B, num_classes)

        return x


def create_network_v2(input_height: int, input_width: int) -> NetworkV2:
    return NetworkV2(input_height, input_width)
