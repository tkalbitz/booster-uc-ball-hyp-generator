from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PySide6.QtCore import QObject, Signal

from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

_logger = get_logger("uc_ball_hyp_generator.labelingtool.sam")

# SAM model URLs and filenames
SAM_MODELS = {
    "sam_vit_h_4b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
    },
    "sam_vit_l_0b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
    },
    "sam_vit_b_01": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
    },
}


class SamManager:
    """Manage SAM model downloading, caching, and inference."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: bool = False
        self._available: bool = False
        self._sam_model: Any = None
        self._predictor: Any = None

    def available(self) -> bool:
        """Return whether a real SAM backend is available."""
        return self._available

    def _download_model(self, progress_cb: Callable[[int], None] | None = None) -> Path:
        """Download the SAM model if not cached."""
        if self._model_name not in SAM_MODELS:
            msg = f"Unsupported SAM model: {self._model_name}"
            raise ValueError(msg)

        model_info = SAM_MODELS[self._model_name]
        model_path = self._cache_dir / model_info["filename"]

        if model_path.exists():
            _logger.info("Using cached SAM model: %s", model_path)
            return model_path

        _logger.info("Downloading SAM model: %s", self._model_name)

        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            if progress_cb is not None and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, int((downloaded / total_size) * 100))
                progress_cb(percent)

        try:
            urllib.request.urlretrieve(model_info["url"], model_path, reporthook=progress_hook)
            _logger.info("Downloaded SAM model to: %s", model_path)
        except Exception as exc:
            if model_path.exists():
                model_path.unlink()
            msg = f"Failed to download SAM model: {exc}"
            raise RuntimeError(msg) from exc

        return model_path

    def _ensure_loaded(self, progress_cb: Callable[[int], None] | None = None) -> None:
        """Ensure SAM model is loaded and ready for inference."""
        if self._loaded:
            return

        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore[import-untyped]

            if progress_cb is not None:
                progress_cb(10)

            # Download model if needed
            model_path = self._download_model(progress_cb)

            if progress_cb is not None:
                progress_cb(70)

            # Load the SAM model
            model_type = self._model_name.split("_")[1]  # Extract 'vit_h', 'vit_l', or 'vit_b'
            if model_type not in sam_model_registry:
                msg = f"Model type {model_type} not found in registry"
                raise ValueError(msg)

            _logger.info("Model type %s found in registry", model_type)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _logger.info("Loading SAM model on device: %s", device)

            self._sam_model = sam_model_registry[model_type](checkpoint=str(model_path))
            self._sam_model.to(device=device)

            self._predictor = SamPredictor(self._sam_model)

            if progress_cb is not None:
                progress_cb(100)

            self._available = True
            _logger.info("SAM model loaded successfully")

        except ImportError as exc:
            _logger.warning("segment-anything not available: %s", exc)
            self._available = False
        except Exception as exc:  # noqa: BLE001
            _logger.error("Failed to load SAM model: %s", exc)
            self._available = False

        self._loaded = True

    def infer(self, image_path: Path, progress_cb: Callable[[int], None] | None = None) -> list[np.ndarray]:
        """Run SAM inference on an image and return segmentation masks."""
        self._ensure_loaded(progress_cb)

        if not self._available or self._predictor is None:
            # Return fallback placeholder masks if SAM is not available
            return self._generate_placeholder_masks(image_path, progress_cb)

        try:
            if progress_cb is not None:
                progress_cb(10)

            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                msg = f"Failed to load image: {image_path}"
                raise RuntimeError(msg)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if progress_cb is not None:
                progress_cb(30)

            # Set image for SAM predictor
            self._predictor.set_image(image_rgb)

            if progress_cb is not None:
                progress_cb(60)

            # Generate masks using automatic mask generation
            try:
                from segment_anything import SamAutomaticMaskGenerator  # type: ignore[import-untyped]

                mask_generator = SamAutomaticMaskGenerator(
                    model=self._sam_model,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )

                if progress_cb is not None:
                    progress_cb(80)

                masks = mask_generator.generate(image_rgb)

                if progress_cb is not None:
                    progress_cb(95)

                # Convert to list of boolean masks
                result_masks = []
                for mask_dict in masks:
                    if "segmentation" in mask_dict:
                        result_masks.append(mask_dict["segmentation"].astype(bool))

                # Sort by area (largest first) and limit to reasonable number
                result_masks.sort(key=lambda m: np.sum(m), reverse=True)
                result_masks = result_masks[:20]  # Limit to top 20 masks

                if progress_cb is not None:
                    progress_cb(100)

                _logger.info("Generated %d masks for image: %s", len(result_masks), image_path)
                return result_masks

            except ImportError:
                _logger.warning("SamAutomaticMaskGenerator not available, using predictor mode")
                # Fallback to point-based prediction
                height, width = image_rgb.shape[:2]

                # Generate some sample points across the image
                points = []
                for y in range(height // 4, height, height // 4):
                    for x in range(width // 4, width, width // 4):
                        points.append([x, y])

                point_coords = np.array(points)
                point_labels = np.ones(len(points))

                masks, _, _ = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

                result_masks = [mask.astype(bool) for mask in masks]

                if progress_cb is not None:
                    progress_cb(100)

                return result_masks

        except Exception as exc:  # noqa: BLE001
            _logger.error("SAM inference failed: %s", exc)
            # Return fallback masks on error
            return self._generate_placeholder_masks(image_path, progress_cb)

    def _generate_placeholder_masks(
        self, image_path: Path, progress_cb: Callable[[int], None] | None = None
    ) -> list[np.ndarray]:
        """Generate placeholder masks when SAM is not available."""
        _logger.info("Using placeholder masks for: %s", image_path)

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                msg = f"Failed to load image: {image_path}"
                raise RuntimeError(msg)

            height, width = image.shape[:2]
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Failed to load image for placeholder masks, using default size: %s", exc)
            width, height = 640, 480

        if progress_cb is not None:
            progress_cb(10)

        # Generate some fake circular masks
        masks = []

        # Central elliptical mask
        cy = height // 2
        cx = width // 2
        ry = max(8, height // 6)
        rx = max(8, width // 6)
        y, x = np.ogrid[:height, :width]
        mask1 = ((x - cx) * (x - cx)) / (rx * rx) + ((y - cy) * (y - cy)) / (ry * ry) <= 1.0
        masks.append(mask1.astype(bool))

        if progress_cb is not None:
            progress_cb(50)

        # Small circular mask in upper left
        r2 = max(6, min(width, height) // 10)
        y0 = r2 + 10
        x0 = r2 + 10
        mask2 = np.zeros((height, width), dtype=bool)
        mask2[((x - x0) * (x - x0) + (y - y0) * (y - y0)) <= r2 * r2] = True
        masks.append(mask2)

        # Small circular mask in lower right
        y0 = height - r2 - 10
        x0 = width - r2 - 10
        mask3 = np.zeros((height, width), dtype=bool)
        mask3[((x - x0) * (x - x0) + (y - y0) * (y - y0)) <= r2 * r2] = True
        masks.append(mask3)

        if progress_cb is not None:
            progress_cb(100)

        return masks

    def get_bounding_box_from_mask(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """Extract tight bounding box coordinates from a mask."""
        if not np.any(mask):
            return (0, 0, 0, 0)

        # Find the bounding box of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max), int(y_max))


class SamWorker(QObject):
    """Run SAM inference in a background thread."""

    progress = Signal(int)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, path: Path, manager: SamManager) -> None:
        super().__init__()
        self._path = path
        self._manager = manager

    def run(self) -> None:
        """Execute SAM inference on the image."""
        try:
            self.progress.emit(0)
            masks = self._manager.infer(self._path, self.progress.emit)
            self.finished.emit(self._path, masks)
        except Exception as exc:  # noqa: BLE001
            msg: str = str(exc)
            self.error.emit(msg)
