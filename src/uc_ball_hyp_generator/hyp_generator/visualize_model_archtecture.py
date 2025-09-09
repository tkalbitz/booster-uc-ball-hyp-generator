#!/usr/bin/env python3

"""Model visualization script for NetworkV2 architecture."""

import torch
from torchinfo import summary

from uc_ball_hyp_generator.hyp_generator import config
from uc_ball_hyp_generator.hyp_generator.model import NetworkV2


def visualize_model(input_height: int = 480, input_width: int = 640, batch_size: int = 1) -> None:
    """Visualize the NetworkV2 model architecture with layer details and dimensions."""
    model = NetworkV2(input_height=input_height, input_width=input_width, num_classes=2)

    input_shape = (batch_size, 3, input_height, input_width)

    print("Model Architecture Visualization")
    print(f"Input Shape: {input_shape}")
    print("=" * 80)

    # Detailed model summary with layer-wise output shapes and parameters
    model_summary = summary(
        model,
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        verbose=2,
        depth=4,
    )

    print("\n" + "=" * 80)
    print("Model Summary:")
    print("=" * 80)
    print(model_summary)


def visualize_forward_pass(input_height: int = 224, input_width: int = 224) -> None:
    """Show tensor shapes during forward pass for debugging."""
    model = NetworkV2(input_height=input_height, input_width=input_width, num_classes=2)
    model.eval()

    # Create dummy input
    x = torch.randn(1, 3, input_height, input_width)

    print("\nForward Pass Tensor Shapes:")
    print("=" * 50)
    print(f"Input shape: {x.shape}")

    # Backbone
    x = model.backbone(x)
    print(f"After backbone: {x.shape}")

    # Feature blocks
    for i, block in enumerate(model.feature_blocks):
        x = block(x)
        print(f"After feature block {i + 1}: {x.shape}")

    # Classifier
    x = model.classifier(x)
    print(f"Final output: {x.shape}")


if __name__ == "__main__":
    # Default input dimensions (can be changed)
    height, width = config.patch_height, config.patch_width

    print(f"Visualizing NetworkV2 with input size ({height}, {width})")

    # Show detailed architecture
    visualize_model(input_height=height, input_width=width)

    # Show forward pass shapes
    visualize_forward_pass(input_height=height, input_width=width)
