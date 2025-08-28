import torch
from torchprofile import profile_macs
from torch import nn


def get_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Calculate FLOPs for PyTorch model."""
    dummy_input = torch.randn(1, *input_shape)
    model.eval()
    with torch.no_grad():
        macs = profile_macs(model, dummy_input)
    return macs * 2  # MACs to FLOPs conversion