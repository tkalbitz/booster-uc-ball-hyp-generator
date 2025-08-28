#!/usr/bin/env python3
"""Compare accuracy between different precision TensorRT models."""

import sys
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np

from inference_tensorrt import TensorRTInference
import models
from config import patch_height, patch_width, image_dir, testset_csv_collection
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset
from scale import unscale_x, unscale_y
from custom_metrics import FoundBallMetric


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


def evaluate_model_accuracy(
    model: Any, 
    data_loader: Any, 
    model_type: str,
    num_samples: int = 1000
) -> dict[str, float]:
    """Evaluate model accuracy on test data.
    
    Args:
        model: Model to evaluate (PyTorch or TensorRT)
        data_loader: Data loader with test samples
        model_type: Type of model ("pytorch", "tensorrt")
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with accuracy metrics
    """
    found_ball_metric = FoundBallMetric()
    total_samples = 0
    total_time = 0.0
    distance_errors = []
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if total_samples >= num_samples:
            break
            
        if model_type == "pytorch":
            images = images.cuda()
            start_time = time.time()
            with torch.no_grad():
                predictions = model(images)
            inference_time = time.time() - start_time
            
            predictions_np = predictions.cpu().numpy()
            
        elif model_type == "tensorrt":
            images_np = images.numpy()
            start_time = time.time()
            predictions_np = model.predict(images_np)
            inference_time = time.time() - start_time
            
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)
        
        total_time += inference_time
        
        labels_np = labels.numpy()
        
        for i in range(len(predictions_np)):
            pred = predictions_np[i]
            true = labels_np[i]
            
            x_pred = unscale_x(pred[0]) + patch_width / 2
            y_pred = unscale_y(pred[1]) + patch_height / 2
            
            x_true = unscale_x(true[0]) + patch_width / 2
            y_true = unscale_y(true[1]) + patch_height / 2
            radius = true[2]
            
            distance = np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)
            distance_errors.append(distance)
            
            found = distance < radius
            found_ball_metric.update(found, 1)
            
            total_samples += 1
            if total_samples >= num_samples:
                break
    
    accuracy = found_ball_metric.result()
    avg_inference_time = (total_time / total_samples) * 1000  # Convert to ms
    mean_distance_error = np.mean(distance_errors)
    std_distance_error = np.std(distance_errors)
    
    return {
        "accuracy": accuracy,
        "avg_inference_time_ms": avg_inference_time,
        "mean_distance_error": mean_distance_error,
        "std_distance_error": std_distance_error,
        "total_samples": total_samples
    }


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
        
        data_loader = create_dataset(test_img, test_labels, 1, trainset=False)
        
    except Exception as e:
        print(f"Warning: Could not load real test data: {e}")
        print("Using synthetic data for comparison...")
        
        def synthetic_data_loader():
            for _ in range(num_samples):
                images = torch.randn(1, 3, patch_height, patch_width)
                labels = torch.tensor([[0.1, 0.1, 10.0]])  # Synthetic label
                yield images, labels
        
        data_loader = synthetic_data_loader()
    
    results = {}
    
    # Evaluate original PyTorch model
    print("1. Evaluating PyTorch FP32 model...")
    try:
        pytorch_model = load_pytorch_model(pytorch_model_path)
        results["pytorch_fp32"] = evaluate_model_accuracy(
            pytorch_model, data_loader, "pytorch", num_samples
        )
        print(f"   ✓ Accuracy: {results['pytorch_fp32']['accuracy']:.4f}")
        print(f"   ✓ Avg time: {results['pytorch_fp32']['avg_inference_time_ms']:.2f} ms")
        
    except Exception as e:
        print(f"   ✗ Failed to evaluate PyTorch model: {e}")
    
    # Evaluate FP16 TensorRT model
    if fp16_model_path and Path(fp16_model_path).exists():
        print("\n2. Evaluating TensorRT FP16 model...")
        try:
            fp16_model = TensorRTInference(fp16_model_path)
            
            data_loader_fp16 = create_dataset(test_img, test_labels, 1, trainset=False) if 'test_img' in locals() else synthetic_data_loader()
            results["tensorrt_fp16"] = evaluate_model_accuracy(
                fp16_model, data_loader_fp16, "tensorrt", num_samples
            )
            print(f"   ✓ Accuracy: {results['tensorrt_fp16']['accuracy']:.4f}")
            print(f"   ✓ Avg time: {results['tensorrt_fp16']['avg_inference_time_ms']:.2f} ms")
            
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
                int8_model, data_loader_int8, "tensorrt", num_samples
            )
            print(f"   ✓ Accuracy: {results['tensorrt_int8']['accuracy']:.4f}")
            print(f"   ✓ Avg time: {results['tensorrt_int8']['avg_inference_time_ms']:.2f} ms")
            
        except Exception as e:
            print(f"   ✗ Failed to evaluate INT8 model: {e}")
    else:
        print(f"\n3. Skipping INT8 evaluation (file not found: {int8_model_path})")
    
    # Print comparison summary
    if len(results) > 1:
        print("\n=== Comparison Summary ===")
        print(f"{'Model':<15} {'Accuracy':<10} {'Time (ms)':<12} {'Speedup':<10} {'Accuracy Loss':<15}")
        print("-" * 70)
        
        baseline_time = results.get("pytorch_fp32", {}).get("avg_inference_time_ms", 1.0)
        baseline_accuracy = results.get("pytorch_fp32", {}).get("accuracy", 1.0)
        
        for model_name, metrics in results.items():
            speedup = baseline_time / metrics["avg_inference_time_ms"]
            accuracy_loss = (baseline_accuracy - metrics["accuracy"]) * 100
            
            print(f"{model_name:<15} {metrics['accuracy']:<10.4f} "
                  f"{metrics['avg_inference_time_ms']:<12.2f} "
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