import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def get_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """
    Calculate FLOPs and other metrics for a PyTorch model using torch.utils.flop_counter.

    Args:
        model: The PyTorch model.
        input_shape: The shape of the input tensor (e.g., (3, 32, 32)).

    Returns:
        A dictionary containing total FLOPs, total MACs, and operator-level details.
    """
    dummy_input = torch.randn(1, *input_shape)
    model.eval()

    # Use FlopCounterMode to count the operations
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(dummy_input)

    return flop_counter.get_total_flops()
