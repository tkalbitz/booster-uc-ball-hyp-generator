import torch
import torch.nn as nn
import torch.nn.functional as F


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
