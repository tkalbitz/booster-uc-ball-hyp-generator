"""Create ball detection statistics CSV.

The script loads a hypothesis model and a classifier model, processes all
positive and negative samples listed in a CSV collection and records binary
classification metrics for a range of thresholds.

Typical usage:
    python -m uc_ball_hyp_generator.visualization.create_ball_statistics \
        --labels /path/to/labels.txt \
        --hyp-weights /path/to/hypothesis.pth \
        --clf-weights /path/to/classifier.pth \
        --image-dir /path/to/images \
        [--output /tmp/ball_classifier.csv] \
        [--start-threshold 0.99] \
        [--step 0.005] \
"""

import argparse
import logging
import os
import queue
import threading
from pathlib import Path

import kornia
import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from torchvision.io import ImageReadMode, decode_image
from tqdm import tqdm

from uc_ball_hyp_generator.classifier.utils import (
    load_ball_classifier_model,
    run_ball_classifier_model,
)
from uc_ball_hyp_generator.hyp_generator.config import (
    img_scaled_height,
    img_scaled_width,
)
from uc_ball_hyp_generator.hyp_generator.utils import (
    BallHypothesis,
    load_ball_hyp_model,
    run_ball_hyp_model,
)
from uc_ball_hyp_generator.utils.binary_classifier_statistics import (
    BinaryClassifierStatistics,
)
from uc_ball_hyp_generator.utils.csv_label_reader import load_csv_collection

_logger = logging.getLogger(__name__)


def _preprocess_image(image_path: Path) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Load an image and return (original_yuv, scaled_yuv, original_size)."""
    img_tensor = decode_image(str(image_path), mode=ImageReadMode.RGB)
    original_h, original_w = img_tensor.shape[1], img_tensor.shape[2]
    original_size = (original_w, original_h)

    img_float = transforms_v2.ToDtype(torch.float32, scale=True)(img_tensor)

    scaled = transforms_v2.Resize((img_scaled_height, img_scaled_width), antialias=True)(img_float)

    original_yuv = kornia.color.rgb_to_yuv(img_float.unsqueeze(0)).squeeze(0)
    scaled_yuv = kornia.color.rgb_to_yuv(scaled.unsqueeze(0)).squeeze(0)

    return original_yuv, scaled_yuv, original_size


def _generate_thresholds(start: float, step: float) -> np.ndarray:
    """Create an array of thresholds from start up to 1.0 inclusive."""
    return np.arange(start, 1.0 + step / 2, step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSV with binary classification statistics for ball detection."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to the CSV file containing image labels (same format as training CSVs).",
    )
    parser.add_argument(
        "--hyp-weights",
        type=Path,
        required=True,
        help="Path to the hypothesis model weights (.pth).",
    )
    parser.add_argument(
        "--clf-weights",
        type=Path,
        required=True,
        help="Path to the classifier model weights (.pth).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing the image files referenced in the CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ball_classifier.csv"),
        help="Output CSV file path (default: /tmp/ball_classifier.csv).",
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.99,
        help="Initial threshold value (default: 0.99).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.00005,
        help="Threshold increment (default: 0.005).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images processed per worker batch (default: 128).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: CPU count).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Using device: %s", device)

    png_files = {f.name: str(f) for f in args.image_dir.glob("**/*.png")}
    pos_imgs, pos_labels, neg_imgs, _ = load_csv_collection(args.labels, png_files)
    pos_bbox_map: dict[str, tuple[int, int, int, int]] = {
        img: bbox for img, bbox in zip(pos_imgs, pos_labels, strict=False)
    }

    all_imgs = pos_imgs + neg_imgs
    all_labels = [1] * len(pos_imgs) + [0] * len(neg_imgs)

    hyp_model = load_ball_hyp_model(args.hyp_weights, device)
    clf_model = load_ball_classifier_model(args.clf_weights, device)

    thresholds = _generate_thresholds(args.start_threshold, args.step)
    stats = BinaryClassifierStatistics(thresholds)
    fp_records: list[str] = []

    total_images = len(all_imgs)

    # Queue of tasks (image path, label)
    task_queue: queue.Queue[tuple[str, int]] = queue.Queue()
    for img_path, label in zip(all_imgs, all_labels, strict=False):
        task_queue.put((img_path, label))

    # Queue of pre‑processed tensors, limited to 128 items to bound memory usage
    result_queue: queue.Queue[tuple[torch.Tensor, torch.Tensor, int, str]] = queue.Queue(maxsize=128)

    def _loader() -> None:
        while True:
            try:
                img_path, label = task_queue.get_nowait()
            except queue.Empty:
                break
            original_yuv, scaled_yuv, _ = _preprocess_image(Path(img_path))
            result_queue.put((original_yuv, scaled_yuv, label, img_path))
            task_queue.task_done()

    workers = args.workers
    threads = [threading.Thread(target=_loader, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()

    processed = 0
    with tqdm(total=total_images, desc="Processing images") as pbar:
        while processed < total_images:
            original_yuv, scaled_yuv, true_label, img_path = result_queue.get()
            original_yuv = original_yuv.to(device)
            scaled_yuv = scaled_yuv.to(device)

            hyps: list[BallHypothesis] = run_ball_hyp_model(hyp_model, scaled_yuv)
            if true_label == 1:
                bbox = pos_bbox_map.get(img_path)
                if bbox:
                    hyps = [
                        h
                        for h in hyps
                        if bbox[0] <= h.center_x <= bbox[0] + bbox[2] and bbox[1] <= h.center_y <= bbox[1] + bbox[3]
                    ]
            if hyps:
                probs: list[float] = run_ball_classifier_model(clf_model, original_yuv, hyps)
                if probs:
                    probas_tensor = torch.tensor(probs, dtype=torch.float32)
                    labels_tensor = torch.full_like(probas_tensor, fill_value=true_label, dtype=torch.float32)
                    stats.add_batch(probas_tensor, labels_tensor)

                    if true_label == 0:
                        for hyp, prob in zip(hyps, probs, strict=False):
                            if prob >= args.start_threshold:
                                fp_records.append(f"{img_path} {hyp.center_x} {hyp.center_y} {hyp.diameter}")

            processed += 1
            pbar.update(1)
            result_queue.task_done()

    task_queue.join()
    for t in threads:
        t.join()

    df = stats.get_results_as_dataframe()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        f.write(df.to_string(float_format="%.6f", index=False))
    _logger.info("Statistics written to %s", args.output)

    fp_output_path = args.output.with_name(args.output.name + ".fp.files.txt")
    fp_output_path.parent.mkdir(parents=True, exist_ok=True)
    with fp_output_path.open("w") as fp_file:
        for line in fp_records:
            fp_file.write(f"{line}\n")
    _logger.info("False positive records written to %s", fp_output_path)


if __name__ == "__main__":
    main()
