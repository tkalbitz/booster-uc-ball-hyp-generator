import torch
import torch.nn as nn

from uc_ball_hyp_generator.layer.conv2dbn import Conv2dBn


class BigLittleReduction(nn.Module):
    def __init__(self, in_channels: int, filter_count: int, activation: str = "relu") -> None:
        super().__init__()
        self.conv1_3x3 = Conv2dBn(
            in_channels, filter_count // 2, kernel_size=3, stride=(2, 2), padding=1, activation=activation
        )
        self.conv2_3x3 = Conv2dBn(in_channels, filter_count, kernel_size=3, stride=1, padding=1, activation=activation)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_out = self.conv1_3x3(x)
        conv2_out = self.conv2_3x3(x)
        conv2_out = self.avg_pool(conv2_out)

        return torch.cat([conv1_out, conv2_out], dim=1)
