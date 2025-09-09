import torch
import torch.nn as nn


class Conv2dBn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_filter: int,
        kernel_size: int,
        stride: int | tuple[int, int] = 1,
        padding: int | str = "same",
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, nb_filter, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nb_filter, affine=False)
        self.activation = self._activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    def _activation(self, act: str) -> nn.Module:
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
