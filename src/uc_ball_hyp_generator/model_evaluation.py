"""Model evaluation utilities for ball detection."""

import time
from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from uc_ball_hyp_generator.config import patch_height, patch_width
from uc_ball_hyp_generator.custom_metrics import FoundBallMetric
from uc_ball_hyp_generator.scale import unscale_x, unscale_y


@runtime_checkable
class PyTorchModel(Protocol):
    """Protocol for PyTorch models."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        ...

    def eval(self) -> torch.nn.Module:
        """Set model to evaluation mode."""
        ...


@runtime_checkable
class TensorRTModel(Protocol):
    """Protocol for TensorRT models."""

    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run inference on input data."""
        ...


@dataclass
class InferenceResult:
    """Result of model inference on a batch."""

    predictions: npt.NDArray[np.float32]
    inference_time: float


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for model performance."""

    accuracy: float
    avg_inference_time_ms: float
    mean_distance_error: float
    std_distance_error: float
    total_samples: int


def run_pytorch_inference(model: PyTorchModel, images: torch.Tensor, device: torch.device) -> InferenceResult:
    """Run inference using PyTorch model."""
    images = images.to(device)
    start_time = time.time()

    with torch.no_grad():
        predictions = model(images)

    inference_time = time.time() - start_time
    predictions_np = predictions.cpu().numpy()

    return InferenceResult(predictions_np, inference_time)


def run_tensorrt_inference(model: TensorRTModel, images: torch.Tensor) -> InferenceResult:
    """Run inference using TensorRT model."""
    images_np = images.numpy()
    start_time = time.time()

    predictions_np = model.predict(images_np)
    inference_time = time.time() - start_time

    return InferenceResult(predictions_np, inference_time)


def process_predictions_batch(
    predictions: npt.NDArray[np.float32], labels: npt.NDArray[np.float32], found_ball_metric: FoundBallMetric
) -> list[float]:
    """Process a batch of predictions and compute distance errors."""
    distance_errors = []

    for i in range(len(predictions)):
        pred = predictions[i]
        true = labels[i]

        x_pred = unscale_x(pred[0]) + patch_width / 2
        y_pred = unscale_y(pred[1]) + patch_height / 2

        x_true = unscale_x(true[0]) + patch_width / 2
        y_true = unscale_y(true[1]) + patch_height / 2
        radius = true[2]

        distance = float(np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2))
        distance_errors.append(distance)

        found = distance < radius
        if found:
            found_ball_metric.found_balls += 1
        found_ball_metric.totals_balls += 1

    return distance_errors


def evaluate_model_accuracy(
    model: PyTorchModel | TensorRTModel,
    data_loader: DataLoader | Iterator[tuple[torch.Tensor, torch.Tensor]],
    model_type: str,
    device: torch.device | None = None,
    num_samples: int = 1000,
) -> EvaluationMetrics:
    """Evaluate model accuracy on test data.

    Args:
        model: Model to evaluate (PyTorch or TensorRT)
        data_loader: Data loader with test samples
        model_type: Type of model ("pytorch", "tensorrt")
        device: Device to run evaluation on (for PyTorch models)
        num_samples: Number of samples to evaluate

    Returns:
        EvaluationMetrics dataclass with accuracy, timing, and error statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    found_ball_metric = FoundBallMetric()
    total_samples = 0
    total_time = 0.0
    distance_errors: list[float] = []

    for images, labels in data_loader:
        if total_samples >= num_samples:
            break

        if model_type == "pytorch":
            if not isinstance(model, PyTorchModel):
                msg = "Model must implement PyTorchModel protocol for pytorch model_type"
                raise TypeError(msg)
            result = run_pytorch_inference(model, images, device)
        elif model_type == "tensorrt":
            if not isinstance(model, TensorRTModel):
                msg = "Model must implement TensorRTModel protocol for tensorrt model_type"
                raise TypeError(msg)
            result = run_tensorrt_inference(model, images)
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        total_time += result.inference_time
        labels_np = labels.numpy()

        batch_errors = process_predictions_batch(result.predictions, labels_np, found_ball_metric)
        distance_errors.extend(batch_errors)

        total_samples += len(result.predictions)
        if total_samples >= num_samples:
            break

    accuracy = found_ball_metric.result()
    avg_inference_time = (total_time / total_samples) * 1000  # Convert to ms
    mean_distance_error = float(np.mean(distance_errors))
    std_distance_error = float(np.std(distance_errors))

    return EvaluationMetrics(
        accuracy=accuracy,
        avg_inference_time_ms=avg_inference_time,
        mean_distance_error=mean_distance_error,
        std_distance_error=std_distance_error,
        total_samples=total_samples,
    )
