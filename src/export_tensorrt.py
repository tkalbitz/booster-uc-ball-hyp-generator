#!/usr/bin/env python3
"""Export PyTorch model to TensorRT engine."""

import sys
from pathlib import Path
from typing import Iterator, Any

import torch
import tensorrt as trt  # type: ignore[import-untyped]
import numpy as np
from torch2trt import torch2trt  # type: ignore[import-not-found]

import models
from config import patch_height, patch_width, image_dir, testset_csv_collection
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator for TensorRT quantization."""
    
    def __init__(self, data_loader: Iterator[torch.Tensor], cache_file: str = "int8_calibration.cache"):
        """Initialize INT8 calibrator.
        
        Args:
            data_loader: Iterator yielding calibration data batches
            cache_file: Path to calibration cache file
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = iter(data_loader)
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_batch = None
        
        self.device_input = None
        
    def get_batch_size(self) -> int:
        """Return the batch size for calibration."""
        return self.batch_size
        
    def get_batch(self, names: list[str]) -> list[int] | None:
        """Get next batch of calibration data.
        
        Args:
            names: List of input tensor names
            
        Returns:
            List of device pointers or None if no more data
        """
        try:
            batch = next(self.data_loader)
            if isinstance(batch, tuple):
                batch = batch[0]  # Take only input data, ignore labels
            
            batch_np = batch.cpu().numpy().astype(np.float32)
            
            if self.device_input is None:
                import pycuda.driver as cuda  # type: ignore[import-not-found]
                self.device_input = cuda.mem_alloc(batch_np.nbytes)
            
            import pycuda.driver as cuda  # type: ignore[import-not-found]
            cuda.memcpy_htod(self.device_input, batch_np.ravel())
            
            return [int(self.device_input)] if self.device_input is not None else None
        except StopIteration:
            return None
            
    def read_calibration_cache(self) -> bytes:
        """Read calibration cache from file."""
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return b''
        
    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache to file."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def _create_calibration_data(num_batches: int, device: torch.device) -> Iterator[torch.Tensor]:
    """Create calibration data for INT8 quantization.
    
    Args:
        num_batches: Number of calibration batches to create
        device: Device to load data on
        
    Yields:
        Calibration data batches
    """
    try:
        png_files = {f.name: str(f) for f in image_dir.glob("**/*.png")}
        test_img, test_labels, _ = load_csv_collection(testset_csv_collection, png_files)
        
        ds = create_dataset(test_img, test_labels, 1, trainset=False)
        
        count = 0
        for batch in ds:
            if count >= num_batches:
                break
            images, _ = batch
            yield images.to(device)
            count += 1
            
    except Exception as e:
        print(f"Warning: Could not load real data for calibration: {e}")
        print("Using synthetic data for calibration...")
        
        for _ in range(num_batches):
            synthetic_data = torch.randn(1, 3, patch_height, patch_width).to(device)
            yield synthetic_data


def create_tensorrt_engine(
    model_path: str,
    output_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    device: str = "cuda",
    calibration_batches: int = 100
) -> None:
    """Export PyTorch model to TensorRT engine.
    
    Args:
        model_path: Path to PyTorch model weights
        output_path: Path to save TensorRT engine
        fp16: Enable FP16 precision for faster inference
        int8: Enable INT8 quantization for maximum speed
        max_batch_size: Maximum batch size for optimization
        device: Device to use for conversion
        calibration_batches: Number of batches for INT8 calibration
    """
    torch_device = torch.device(device)
    
    model = models.create_network_v2(patch_height, patch_width)
    model.load_state_dict(torch.load(model_path, map_location=torch_device))
    model.to(torch_device)
    model.eval()
    
    dummy_input = torch.randn(max_batch_size, 3, patch_height, patch_width).to(torch_device)
    
    print("Converting model to TensorRT...")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Using FP16: {fp16}")
    print(f"Using INT8: {int8}")
    print(f"Max batch size: {max_batch_size}")
    
    conversion_kwargs: dict[str, Any] = {
        "fp16_mode": fp16,
        "max_batch_size": max_batch_size
    }
    
    if int8:
        print(f"Preparing INT8 calibration data ({calibration_batches} batches)...")
        calibration_data = _create_calibration_data(calibration_batches, torch_device)
        conversion_kwargs.update({
            "int8_mode": True,
            "int8_calib_dataset": calibration_data
        })
    
    with torch.no_grad():
        model_trt = torch2trt(
            model, 
            [dummy_input],
            **conversion_kwargs
        )
    
    torch.save(model_trt.state_dict(), output_path)
    print(f"TensorRT model saved to: {output_path}")
    
    with torch.no_grad():
        y_torch = model(dummy_input)
        y_trt = model_trt(dummy_input)
        
    error = torch.max(torch.abs(y_torch - y_trt))
    print(f"Max error between PyTorch and TensorRT: {error.item():.6f}")


def create_trt_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_size: int = 1 << 30,
    calibration_batches: int = 100
) -> None:
    """Create TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        int8: Enable INT8 quantization
        max_batch_size: Maximum batch size
        workspace_size: Maximum workspace size in bytes
        calibration_batches: Number of batches for INT8 calibration
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        calibration_data = _create_calibration_data(calibration_batches, torch.device('cuda'))
        calibrator = Int8Calibrator(calibration_data, f"{Path(engine_path).stem}_calibration.cache")
        config.int8_calibrator = calibrator
        
    profile = builder.create_optimization_profile()
    
    input_tensor = network.get_input(0)
    input_shape = (max_batch_size, 3, patch_height, patch_width)
    
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    
    print("Building TensorRT engine from ONNX...")
    print(f"Input shape: {input_shape}")
    print(f"Using FP16: {fp16}")
    print(f"Using INT8: {int8}")
    if int8:
        print(f"INT8 calibration batches: {calibration_batches}")
    print(f"Workspace size: {workspace_size / (1024**3):.1f} GB")
    
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("Failed to build TensorRT engine")
        return
        
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to: {engine_path}")


def main() -> None:
    """Main function to handle command line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python export_tensorrt.py <model_path> <output_path> [options]")
        print("Options:")
        print("  --fp16                Enable FP16 precision (default: True)")
        print("  --int8                Enable INT8 quantization for maximum speed")
        print("  --no-fp16             Disable FP16 precision")
        print("  --batch-size N        Maximum batch size (default: 1)")
        print("  --from-onnx           Convert from ONNX instead of PyTorch")
        print("  --workspace-size N    Workspace size in GB for ONNX conversion (default: 1)")
        print("  --calibration-batches Number of batches for INT8 calibration (default: 100)")
        return
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    fp16 = True
    int8 = False
    max_batch_size = 1
    from_onnx = False
    workspace_gb = 1.0
    calibration_batches = 100
    
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--fp16":
            fp16 = True
        elif arg == "--no-fp16":
            fp16 = False
        elif arg == "--int8":
            int8 = True
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            max_batch_size = int(sys.argv[i + 1])
            i += 1
        elif arg == "--from-onnx":
            from_onnx = True
        elif arg == "--workspace-size" and i + 1 < len(sys.argv):
            workspace_gb = float(sys.argv[i + 1])
            i += 1
        elif arg == "--calibration-batches" and i + 1 < len(sys.argv):
            calibration_batches = int(sys.argv[i + 1])
            i += 1
        i += 1
    
    if not Path(model_path).exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)
    
    try:
        if from_onnx:
            workspace_size = int(workspace_gb * (1024**3))
            create_trt_engine_from_onnx(
                model_path, 
                output_path, 
                fp16=fp16,
                int8=int8,
                max_batch_size=max_batch_size,
                workspace_size=workspace_size,
                calibration_batches=calibration_batches
            )
        else:
            create_tensorrt_engine(
                model_path, 
                output_path, 
                fp16=fp16,
                int8=int8,
                max_batch_size=max_batch_size,
                calibration_batches=calibration_batches
            )
    except Exception as e:
        error_msg = f"Failed to export model to TensorRT: {e}"
        print(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()