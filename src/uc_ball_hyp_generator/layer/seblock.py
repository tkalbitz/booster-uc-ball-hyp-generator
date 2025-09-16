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
        # Replaced .view() with the more explicit torch.flatten() for clarity.
        # The '1' indicates to start flattening from the first dimension (the channels),
        # preserving the batch dimension.
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        # This reshape is necessary to broadcast the channel multipliers back to the input tensor shape
        x = x.view(x.size(0), x.size(1), 1, 1)
        return input_tensor * x
