#!/usr/bin/env python3
"""Compare accuracy between different precision TensorRT models."""

import sys
from pathlib import Path
from typing import Iterator, Union

import torch
from torch.utils.data import DataLoader

from inference_tensorrt import TensorRTInference
import models
from config import patch_height, patch_width, image_dir, testset_csv_collection
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset
from model_evaluation import evaluate_model_accuracy, EvaluationMetrics


def load_pytorch_model(model_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load original PyTorch model.
    
    Args:
        model_path: Path to PyTorch model weights
        device: Device to load model on
        
    Returns:
        Loaded PyTorch model
    """
    torch_device = torch.device(device)
    model = models.create_network_v2(patch_height, patch_width)
    model.load_state_dict(torch.load(model_path, map_location=torch_device))
    model.to(torch_device)
    model.eval()
    return model




def compare_model_precisions(
    pytorch_model_path: str,
    fp16_model_path: str | None = None,
    int8_model_path: str | None = None,
    num_samples: int = 500
) -> None:
    """Compare accuracy between different precision models.
    
    Args:
        pytorch_model_path: Path to original PyTorch model
        fp16_model_path: Path to FP16 TensorRT model
        int8_model_path: Path to INT8 TensorRT model
        num_samples: Number of samples to evaluate
    """
    print("=== Precision Accuracy Comparison ===")
    print(f"Evaluating on {num_samples} samples...\n")
    
    try:
        png_files = {f.name: str(f) for f in image_dir.glob("**/*.png")}
        test_img, test_labels, _ = load_csv_collection(testset_csv_collection, png_files)
        
        data_loader: Union[DataLoader, Iterator[tuple[torch.Tensor, torch.Tensor]]] = create_dataset(test_img, test_labels, 1, trainset=False)
        
    except Exception as e:
        print(f"Warning: Could not load real test data: {e}")
        print("Using synthetic data for comparison...")
        
        def synthetic_data_loader() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for _ in range(num_samples):
                images = torch.randn(1, 3, patch_height, patch_width)
                labels = torch.tensor([[0.1, 0.1, 10.0]])  # Synthetic label
                yield images, labels
        
        data_loader = synthetic_data_loader()
    
    results: dict[str, EvaluationMetrics] = {}
    
    # Evaluate original PyTorch model
    print("1. Evaluating PyTorch FP32 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        pytorch_model = load_pytorch_model(pytorch_model_path, device.type)
        results["pytorch_fp32"] = evaluate_model_accuracy(
            pytorch_model, data_loader, "pytorch", device, num_samples
        )
        print(f"   ✓ Accuracy: {results['pytorch_fp32'].accuracy:.4f}")
        print(f"   ✓ Avg time: {results['pytorch_fp32'].avg_inference_time_ms:.2f} ms")
        
    except Exception as e:
        print(f"   ✗ Failed to evaluate PyTorch model: {e}")
    
    # Evaluate FP16 TensorRT model
    if fp16_model_path and Path(fp16_model_path).exists():
        print("\n2. Evaluating TensorRT FP16 model...")
        try:
            fp16_model = TensorRTInference(fp16_model_path)
            
            data_loader_fp16 = create_dataset(test_img, test_labels, 1, trainset=False) if 'test_img' in locals() else synthetic_data_loader()
            results["tensorrt_fp16"] = evaluate_model_accuracy(
                fp16_model, data_loader_fp16, "tensorrt", device, num_samples
            )
            print(f"   ✓ Accuracy: {results['tensorrt_fp16'].accuracy:.4f}")
            print(f"   ✓ Avg time: {results['tensorrt_fp16'].avg_inference_time_ms:.2f} ms")
            
        except Exception as e:
            print(f"   ✗ Failed to evaluate FP16 model: {e}")
    else:
        print(f"\n2. Skipping FP16 evaluation (file not found: {fp16_model_path})")
    
    # Evaluate INT8 TensorRT model
    if int8_model_path and Path(int8_model_path).exists():
        print("\n3. Evaluating TensorRT INT8 model...")
        try:
            int8_model = TensorRTInference(int8_model_path)
            
            data_loader_int8 = create_dataset(test_img, test_labels, 1, trainset=False) if 'test_img' in locals() else synthetic_data_loader()
            results["tensorrt_int8"] = evaluate_model_accuracy(
                int8_model, data_loader_int8, "tensorrt", device, num_samples
            )
            print(f"   ✓ Accuracy: {results['tensorrt_int8'].accuracy:.4f}")
            print(f"   ✓ Avg time: {results['tensorrt_int8'].avg_inference_time_ms:.2f} ms")
            
        except Exception as e:
            print(f"   ✗ Failed to evaluate INT8 model: {e}")
    else:
        print(f"\n3. Skipping INT8 evaluation (file not found: {int8_model_path})")
    
    # Print comparison summary
    if len(results) > 1:
        print("\n=== Comparison Summary ===")
        print(f"{'Model':<15} {'Accuracy':<10} {'Time (ms)':<12} {'Speedup':<10} {'Accuracy Loss':<15}")
        print("-" * 70)
        
        baseline_metrics = results.get("pytorch_fp32")
        baseline_time = baseline_metrics.avg_inference_time_ms if baseline_metrics else 1.0
        baseline_accuracy = baseline_metrics.accuracy if baseline_metrics else 1.0
        
        for model_name, metrics in results.items():
            speedup = baseline_time / metrics.avg_inference_time_ms
            accuracy_loss = (baseline_accuracy - metrics.accuracy) * 100
            
            print(f"{model_name:<15} {metrics.accuracy:<10.4f} "
                  f"{metrics.avg_inference_time_ms:<12.2f} "
                  f"{speedup:<10.2f}x {accuracy_loss:<15.2f}%")


def main() -> None:
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python compare_precision_accuracy.py <pytorch_model> [fp16_model] [int8_model]")
        print("Example: python compare_precision_accuracy.py model.pth model_fp16.pth model_int8.pth")
        return
    
    pytorch_model_path = sys.argv[1]
    fp16_model_path = sys.argv[2] if len(sys.argv) > 2 else None
    int8_model_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    compare_model_precisions(
        pytorch_model_path=pytorch_model_path,
        fp16_model_path=fp16_model_path,
        int8_model_path=int8_model_path,
        num_samples=500
    )


if __name__ == "__main__":
    main()