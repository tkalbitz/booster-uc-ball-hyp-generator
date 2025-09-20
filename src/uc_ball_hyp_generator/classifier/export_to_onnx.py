"""
Optimized Export Script for the Ball Classifier Model (PyTorch 2.8+).

This script leverages torch.compile() for pre-export optimization and the modern
torch.export API for a more robust conversion to ONNX with 'channels_last' format.

Example Usage:
    python uc_ball_hyp_generator/classifier/export_to_onnx.py \
        --weights "/path/to/your/classifier/model.acc-XXX.pth" \
        --output "/path/to/output/classifier_model_nhwc.onnx"
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

from uc_ball_hyp_generator.classifier.config import CPATCH_SIZE
from uc_ball_hyp_generator.classifier.model import get_ball_classifier_model
from uc_ball_hyp_generator.utils.common_model_operations import clean_compiled_state_dict


def main() -> None:
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export Classifier Model to ONNX with Channels Last format.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the input PyTorch model weights (.pth).")
    parser.add_argument("--output", type=Path, required=True, help="Path for the output ONNX model.")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found at: {args.weights}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading classifier model with weights from: {args.weights}")

    # 1. Load the model and weights
    model = get_ball_classifier_model()
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(clean_compiled_state_dict(state_dict))
    model.eval()

    # 2. Convert to channels_last for NHWC format
    print("Converting model to channels_last (NHWC) memory format...")
    model.to(memory_format=torch.channels_last)

    # 3. Create a dummy input
    dummy_input = torch.randn(1, 3, CPATCH_SIZE, CPATCH_SIZE).to(memory_format=torch.channels_last)

    # 4. ✅ FINAL FIX: Revert to the stable legacy ONNX exporter.
    print(f"Exporting model with legacy tracer to ONNX at: {args.output}")
    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "N"},
            "output": {0: "N"},
        },
    )

    print("✅ Export complete.")


if __name__ == "__main__":
    main()
