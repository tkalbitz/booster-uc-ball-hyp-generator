import torch
import torch.nn as nn
import torch.nn.functional as F

from uc_ball_hyp_generator.layer.conv2dbn import Conv2dBn


class BigLittleReduction(nn.Module):
    def __init__(self, in_channels: int, filter_count: int, activation: str = "relu") -> None:
        super().__init__()
        self.conv1_3x3 = Conv2dBn(
            in_channels, filter_count // 2, kernel_size=3, stride=(2, 2), padding=1, activation=activation
        )
        self.conv2_3x3 = Conv2dBn(
            in_channels, filter_count, kernel_size=3, stride=1, padding="valid", activation=activation
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_out = self.conv1_3x3(x)
        conv2_out = self.conv2_3x3(x)
        conv2_out = self.avg_pool(conv2_out)

        # Ensure both tensors have the same spatial dimensions for concatenation
        target_h, target_w = conv1_out.shape[2], conv1_out.shape[3]
        current_h, current_w = conv2_out.shape[2], conv2_out.shape[3]

        # Pad conv2_out if necessary to match conv1_out dimensions
        if current_h != target_h or current_w != target_w:
            pad_h = max(0, target_h - current_h)
            pad_w = max(0, target_w - current_w)
            # Pad format: (pad_left, pad_right, pad_top, pad_bottom)
            conv2_out = F.pad(conv2_out, (0, pad_w, 0, pad_h), mode="constant", value=0)

        return torch.cat([conv1_out, conv2_out], dim=1)
