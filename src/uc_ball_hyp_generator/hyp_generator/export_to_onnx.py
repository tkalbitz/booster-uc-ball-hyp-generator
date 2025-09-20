"""
Optimized Export Script for the Ball Hypothesis Model (PyTorch 2.8+).

This script leverages torch.compile() for pre-export optimization and the modern
torch.export API for a more robust conversion to ONNX with 'channels_last' format.

Example Usage:
    python uc_ball_hyp_generator/hyp_generator/export_to_onnx.py \
        --weights "/path/to/your/model/weights.balls.coord.XXX.pth" \
        --output "/path/to/output/hypothesis_model_nhwc.onnx"
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

from uc_ball_hyp_generator.hyp_generator.config import patch_height, patch_width
from uc_ball_hyp_generator.hyp_generator.model import get_ball_hyp_model
from uc_ball_hyp_generator.utils.common_model_operations import clean_compiled_state_dict


def main() -> None:
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export Hypothesis Model to ONNX with Channels Last format.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the input PyTorch model weights (.pth).")
    parser.add_argument("--output", type=Path, required=True, help="Path for the output ONNX model.")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found at: {args.weights}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading hypothesis model with weights from: {args.weights}")

    # 1. Load the model and weights
    model = get_ball_hyp_model(input_height=patch_height, input_width=patch_width)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(clean_compiled_state_dict(state_dict))
    model.eval()

    # 2. Convert to channels_last for NHWC format
    print("Converting model to channels_last (NHWC) memory format...")
    model.to(memory_format=torch.channels_last)

    # 3. Create a dummy input
    dummy_input = torch.randn(1, 3, patch_height, patch_width).to(memory_format=torch.channels_last)

    # 4. ✅ SIMPLIFIED EXPORT: Use the unified torch.onnx.export with the dynamo backend.
    print(f"Exporting model with Dynamo backend to ONNX at: {args.output}")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(args.output),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        # The 'dynamic_axes' format is used by this unified API
        dynamic_axes={
            "input": {0: "N"},
            "output": {0: "N"},
        },
        dynamo=True,  # This flag activates the torch.export backend
    )

    print("✅ Export complete.")


if __name__ == "__main__":
    main()
