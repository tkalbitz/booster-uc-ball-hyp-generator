from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class MetricResult:
    """Stores the calculated metrics for a single threshold."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1_score: float


class BinaryClassifierStatistics:
    """
    A stateful evaluator to accumulate PyTorch batch results and calculate
    binary classification metrics across a range of thresholds.
    """

    def __init__(self, thresholds: list[float] | np.ndarray):
        """Initialize the evaluator with a set of thresholds."""
        if not isinstance(thresholds, (list, np.ndarray)) or len(thresholds) == 0:
            msg = "Thresholds must be a non-empty list or numpy array."
            raise ValueError(msg)
        self.thresholds: np.ndarray = np.array(thresholds)
        self.all_probas_pred: list[np.ndarray] = []
        self.all_y_true: list[np.ndarray] = []

    def add_batch(self, probas_pred_tensor: torch.Tensor, y_true_tensor: torch.Tensor) -> None:
        """Add a batch of predictions and labels to the evaluator."""
        if probas_pred_tensor.is_cuda:
            probas_pred_tensor = probas_pred_tensor.cpu()

        if y_true_tensor.is_cuda:
            y_true_tensor = y_true_tensor.cpu()

        self.all_probas_pred.append(probas_pred_tensor.detach().numpy())
        self.all_y_true.append(y_true_tensor.detach().numpy())

    def _calculate_metrics(self) -> list[MetricResult]:
        """Compute metrics across all accumulated batches."""
        if not self.all_y_true:
            return []

        y_true = np.concatenate(self.all_y_true)
        probas_pred = np.concatenate(self.all_probas_pred)

        results: list[MetricResult] = []
        for thres in self.thresholds:
            y_pred = (probas_pred >= thres).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append(
                MetricResult(
                    threshold=round(thres, 4),
                    tp=int(tp),
                    fp=int(fp),
                    fn=int(fn),
                    tn=int(tn),
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    f1_score=round(f1, 4),
                )
            )
        return results

    def get_results_as_dataframe(self) -> pd.DataFrame:
        """Return the calculated metrics in a pandas DataFrame."""
        if not (metrics := self._calculate_metrics()):
            return pd.DataFrame()

        return pd.DataFrame([asdict(result) for result in metrics])

    def get_results_as_dicts(self) -> list[dict[str, float | int]]:
        """Return the calculated metrics as a list of dictionaries."""
        if not (metrics := self._calculate_metrics()):
            return []

        return [asdict(result) for result in metrics]

    def reset(self) -> None:
        """Reset the evaluator's state, clearing all accumulated batches."""
        self.all_probas_pred = []
        self.all_y_true = []
