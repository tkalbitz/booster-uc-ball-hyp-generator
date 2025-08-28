import torch
from torchprofile import profile_macs


def get_flops(model, input_shape):
    """Calculate FLOPs for PyTorch model."""
    dummy_input = torch.randn(1, *input_shape)
    model.eval()
    with torch.no_grad():
        macs = profile_macs(model, dummy_input)
    return macs * 2  # MACs to FLOPs conversion