#!/usr/bin/env python3
"""Launch script for the ball detection visualizer.

This script launches the naoteamhtwk_machinelearning_visualizer package which handles
all GUI creation and interaction. It simply configures the environment and calls
the visualizer's run function.
"""

import sys
from pathlib import Path

import naoteamhtwk_machinelearning_visualizer as visualizer

# Import the adapter module to set classifier weights
from uc_ball_hyp_generator.visualization import ball_detection_adapter


def main() -> None:
    """Launch the ball detection visualizer."""
    # Check command line arguments for input directory and model path
    if len(sys.argv) not in [3, 4]:
        print("Usage: python launch_ball_visualizer.py <input_directory> <model_path.pth> [classifier_weights.pth]")
        print("Example: python launch_ball_visualizer.py /path/to/images model.pth")
        print("Example with classifier: python launch_ball_visualizer.py /path/to/images model.pth classifier.pth")
        sys.exit(1)

    input_directory = sys.argv[1]
    hyp_model_path = sys.argv[2]
    classifier_weights_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate inputs
    input_path = Path(input_directory)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory does not exist or is not a directory: {input_directory}")
        sys.exit(1)

    hyp_model_file = Path(hyp_model_path)
    if not hyp_model_file.exists():
        print(f"Error: Model file does not exist: {hyp_model_path}")
        sys.exit(1)

    if hyp_model_file.suffix.lower() != ".pth":
        print(f"Error: Model file must have .pth extension: {hyp_model_path}")
        sys.exit(1)

    ball_detection_adapter._hyp_model_path = hyp_model_file.resolve()

    # Validate classifier weights if provided
    if classifier_weights_path:
        classifier_file = Path(classifier_weights_path)
        if not classifier_file.exists():
            print(f"Error: Classifier weights file does not exist: {classifier_weights_path}")
            sys.exit(1)
        if classifier_file.suffix.lower() != ".pth":
            print(f"Error: Classifier weights file must have .pth extension: {classifier_weights_path}")
            sys.exit(1)
        # Set classifier weights path in the adapter module
        ball_detection_adapter._classifier_model_path = classifier_file.resolve()

    print("Launching ball detection visualizer...")
    print(f"Input directory: {input_path.resolve()}")
    print(f"Model file: {hyp_model_file.resolve()}")
    if classifier_weights_path:
        print(f"Classifier weights: {classifier_file.resolve()}")

    try:
        # Launch the visualizer in interactive GUI mode
        # The visualizer package will handle all GUI creation and interaction
        result = visualizer.run_visualizer(
            input_path=input_path,
            output_path=None,  # Interactive mode (no output path = GUI mode)
            adapter_function=None,  # Will be loaded from config
            config_path=None,
            config_overrides={"adapter": {"function": "uc_ball_hyp_generator.visualization.ball_detection_adapter"}},
        )

        if result.success:
            print("Visualizer completed successfully.")
        else:
            print("Visualizer completed with errors:")
            for error in result.errors:
                print(f"  - {error}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nVisualizer interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error launching visualizer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
