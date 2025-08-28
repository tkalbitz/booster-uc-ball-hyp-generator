#!/usr/bin/env python3
"""Run inference with TensorRT model."""

import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from config import patch_height, patch_width
from dataset_handling import yuv2rgb
from scale import unscale_x, unscale_y


class TensorRTInference:
    """TensorRT inference engine."""
    
    def __init__(self, engine_path: str):
        """Initialize TensorRT inference engine.
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        self.input_shape = (1, 3, patch_height, patch_width)
        self.output_shape = (1, 3)
        
        self.input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_size = np.prod(self.output_shape) * np.dtype(np.float32).itemsize
        
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        self.bindings = [int(self.d_input), int(self.d_output)]
        
        self.stream = cuda.Stream()
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data.
        
        Args:
            input_data: Input tensor as numpy array
            
        Returns:
            Prediction results as numpy array
        """
        input_data = input_data.astype(np.float32).ravel()
        
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
        
        self.stream.synchronize()
        
        return output_data


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for inference.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image as numpy array
    """
    image = Image.open(image_path)
    
    if image.size != (patch_width, patch_height):
        image = image.resize((patch_width, patch_height), Image.Resampling.LANCZOS)
    
    image_array = np.array(image)
    
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    
    image_array = image_array.astype(np.float32) / 255.0
    
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor.numpy()


def benchmark_inference(engine_path: str, num_iterations: int = 1000) -> None:
    """Benchmark TensorRT inference speed.
    
    Args:
        engine_path: Path to TensorRT engine
        num_iterations: Number of inference iterations for benchmarking
    """
    trt_engine = TensorRTInference(engine_path)
    
    dummy_input = np.random.randn(*trt_engine.input_shape).astype(np.float32)
    
    warmup_iterations = 10
    for _ in range(warmup_iterations):
        _ = trt_engine.predict(dummy_input)
    
    start_time = time.time()
    for _ in range(num_iterations):
        _ = trt_engine.predict(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000
    fps = 1000 / avg_time
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


def run_inference_on_image(engine_path: str, image_path: str) -> None:
    """Run inference on a single image.
    
    Args:
        engine_path: Path to TensorRT engine
        image_path: Path to input image
    """
    trt_engine = TensorRTInference(engine_path)
    
    input_data = load_and_preprocess_image(image_path)
    
    start_time = time.time()
    prediction = trt_engine.predict(input_data)
    inference_time = (time.time() - start_time) * 1000
    
    x_pred = unscale_x(prediction[0, 0]) + patch_width / 2
    y_pred = unscale_y(prediction[0, 1]) + patch_height / 2
    confidence = prediction[0, 2]
    
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Predicted ball position: ({x_pred:.1f}, {y_pred:.1f})")
    print(f"Confidence: {confidence:.3f}")


def main() -> None:
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python inference_tensorrt.py <engine_path> [options]")
        print("Options:")
        print("  --image <path>        Run inference on single image")
        print("  --benchmark <iters>   Benchmark inference speed (default: 1000)")
        return
    
    engine_path = sys.argv[1]
    
    if not Path(engine_path).exists():
        msg = f"Engine file not found: {engine_path}"
        raise FileNotFoundError(msg)
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "--image" and len(sys.argv) > 3:
            image_path = sys.argv[3]
            run_inference_on_image(engine_path, image_path)
        elif sys.argv[2] == "--benchmark":
            iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
            benchmark_inference(engine_path, iterations)
        else:
            print("Unknown option. Use --image <path> or --benchmark <iterations>")
    else:
        benchmark_inference(engine_path)


if __name__ == "__main__":
    main()