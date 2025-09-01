#!/usr/bin/env python3
"""Example usage of TensorRT export and inference."""

from pathlib import Path

from uc_ball_hyp_generator.export_tensorrt import create_tensorrt_engine, create_trt_engine_from_onnx
from uc_ball_hyp_generator.inference_tensorrt import benchmark_inference


def example_export_workflow() -> None:
    """Demonstrate the complete TensorRT export workflow."""

    model_path = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-22-08-48-01/weights.balls.264-0.954385-1.292296.pth"

    deploy_dir = Path("/tmp/tensorrt_models")
    deploy_dir.mkdir(exist_ok=True)

    print("=== TensorRT Export Example ===")

    # Method 1: Direct PyTorch to TensorRT conversion (FP16)
    print("\n1. Converting PyTorch model directly to TensorRT (FP16)...")
    trt_fp16_path = deploy_dir / "ball_detector_fp16.pth"

    try:
        create_tensorrt_engine(
            model_path=str(model_path), output_path=str(trt_fp16_path), fp16=True, int8=False, max_batch_size=1
        )
        print(f"✓ FP16 TensorRT model saved to: {trt_fp16_path}")

        print("\n2. Benchmarking FP16 TensorRT inference...")
        benchmark_inference(str(trt_fp16_path), num_iterations=100)

    except Exception as e:
        print(f"✗ FP16 conversion failed: {e}")
        print("This might happen if torch2trt is not compatible with your model.")

    # Method 1b: Direct PyTorch to TensorRT conversion (INT8)
    print("\n2b. Converting PyTorch model directly to TensorRT (INT8)...")
    trt_int8_path = deploy_dir / "ball_detector_int8.pth"

    try:
        create_tensorrt_engine(
            model_path=str(model_path),
            output_path=str(trt_int8_path),
            fp16=True,  # Enable both FP16 and INT8 for best performance
            int8=True,
            max_batch_size=1,
            calibration_batches=50,  # Fewer batches for demo
        )
        print(f"✓ INT8 TensorRT model saved to: {trt_int8_path}")

        print("\n3. Benchmarking INT8 TensorRT inference...")
        benchmark_inference(str(trt_int8_path), num_iterations=100)

    except Exception as e:
        print(f"✗ INT8 conversion failed: {e}")
        print("INT8 requires calibration data and may not work with synthetic data.")

    # Method 2: PyTorch -> ONNX -> TensorRT conversion (FP16)
    print("\n4. Converting via ONNX (FP16)...")

    onnx_path = deploy_dir / "ball_detector.onnx"
    engine_fp16_path = deploy_dir / "ball_detector_fp16.engine"
    engine_int8_path = deploy_dir / "ball_detector_int8.engine"

    # First export to ONNX (you can use the existing deploy.py for this)
    print("   Export to ONNX first using deploy.py")
    print("   Then convert ONNX to TensorRT engine...")

    if onnx_path.exists():
        try:
            create_trt_engine_from_onnx(
                onnx_path=str(onnx_path),
                engine_path=str(engine_fp16_path),
                fp16=True,
                int8=False,
                max_batch_size=1,
                workspace_size=1 << 30,  # 1GB
            )
            print(f"✓ FP16 TensorRT engine saved to: {engine_fp16_path}")

            print("\n5. Benchmarking ONNX->TensorRT FP16 inference...")
            benchmark_inference(str(engine_fp16_path), num_iterations=100)

        except Exception as e:
            print(f"✗ ONNX->TensorRT FP16 conversion failed: {e}")

        # Method 2b: ONNX -> TensorRT conversion (INT8)
        try:
            print("\n6. Converting ONNX to TensorRT (INT8)...")
            create_trt_engine_from_onnx(
                onnx_path=str(onnx_path),
                engine_path=str(engine_int8_path),
                fp16=True,
                int8=True,
                max_batch_size=1,
                workspace_size=1 << 30,  # 1GB
                calibration_batches=50,
            )
            print(f"✓ INT8 TensorRT engine saved to: {engine_int8_path}")

            print("\n7. Benchmarking ONNX->TensorRT INT8 inference...")
            benchmark_inference(str(engine_int8_path), num_iterations=100)

        except Exception as e:
            print(f"✗ ONNX->TensorRT INT8 conversion failed: {e}")
    else:
        print(f"   ONNX file not found at {onnx_path}")
        print("   Run deploy.py first to create the ONNX model")


def print_usage_instructions() -> None:
    """Print detailed usage instructions."""
    print("\n=== TensorRT Usage Instructions ===")
    print()
    print("1. Export your PyTorch model to TensorRT (FP16):")
    print("   python export_tensorrt.py model.pth output_fp16.pth --fp16 --batch-size 1")
    print()
    print("2. Export your PyTorch model to TensorRT (INT8 - maximum speed):")
    print("   python export_tensorrt.py model.pth output_int8.pth --int8 --calibration-batches 100")
    print()
    print("3. Alternative: Export via ONNX (more compatible):")
    print("   python deploy.py  # This creates ONNX model")
    print("   python export_tensorrt.py model.onnx output_fp16.engine --from-onnx --fp16")
    print("   python export_tensorrt.py model.onnx output_int8.engine --from-onnx --int8")
    print()
    print("4. Run inference:")
    print("   python inference_tensorrt.py output_fp16.pth --benchmark 1000")
    print("   python inference_tensorrt.py output_int8.pth --image path/to/image.png")
    print()
    print("5. Performance comparison (typical expectations):")
    print("   - PyTorch (CPU):     ~50-100 ms per inference")
    print("   - PyTorch (GPU):     ~5-20 ms per inference")
    print("   - TensorRT FP16:     ~1-5 ms per inference")
    print("   - TensorRT INT8:     ~0.5-2 ms per inference (fastest)")
    print()
    print("6. Accuracy vs Speed trade-offs:")
    print("   - FP32: Best accuracy, slowest")
    print("   - FP16: ~99.5% accuracy, 2x faster")
    print("   - INT8: ~98-99% accuracy, 4-8x faster (requires calibration)")
    print()
    print("Dependencies to install:")
    print("   pip install torch2trt tensorrt pycuda")
    print("   Or with conda: conda install -c nvidia tensorrt")


if __name__ == "__main__":
    print_usage_instructions()
    example_export_workflow()
    example_export_workflow()
