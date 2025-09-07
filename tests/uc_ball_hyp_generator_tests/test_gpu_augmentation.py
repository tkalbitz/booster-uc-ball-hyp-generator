"""Comprehensive tests for GPU-based data augmentation optimization."""

import torch
from torch import Tensor
import pytest

from uc_ball_hyp_generator.dataset_handling import (
    augment_image_test_mode,
    augment_image_test_mode_gpu,
    train_augment_image,
    train_augment_image_gpu,
    train_augment_batch_gpu,
    augment_batch_test_mode_gpu,
    BallDatasetGPU,
)


class TestGPUAugmentationEquivalence:
    """Test that GPU augmentation produces equivalent results to CPU augmentation."""

    @pytest.fixture
    def rgb_chw_single(self) -> Tensor:
        """Create test single RGB image in CHW format."""
        torch.manual_seed(42)
        return torch.rand(3, 30, 40, dtype=torch.float32)

    @pytest.fixture
    def label_single(self) -> Tensor:
        """Create test single label."""
        return torch.tensor([15.0, 20.0, 5.0])

    @pytest.fixture
    def rgb_chw_batch(self) -> Tensor:
        """Create test RGB batch in CHW format."""
        torch.manual_seed(42)
        return torch.rand(8, 3, 30, 40, dtype=torch.float32)

    @pytest.fixture
    def labels_batch(self) -> Tensor:
        """Create test labels batch."""
        torch.manual_seed(123)
        return torch.rand(8, 3) * 20 + 5  # Random labels in reasonable range

    def test_test_augmentation_cpu_gpu_equivalence(self, rgb_chw_single: Tensor, label_single: Tensor) -> None:
        """Test that GPU test augmentation produces same results as CPU version."""
        # CPU version
        yuv_cpu, label_cpu = augment_image_test_mode(rgb_chw_single.clone(), label_single.clone())

        # GPU version (on CPU device for comparison)
        device = torch.device("cpu")
        yuv_gpu, label_gpu = augment_image_test_mode_gpu(rgb_chw_single.clone(), label_single.clone(), device)

        # Results should be nearly identical
        max_diff_image = torch.max(torch.abs(yuv_cpu - yuv_gpu))
        max_diff_label = torch.max(torch.abs(label_cpu - label_gpu))

        assert max_diff_image < 1e-6, f"Test augmentation image difference {max_diff_image} exceeds tolerance"
        assert max_diff_label < 1e-6, f"Test augmentation label difference {max_diff_label} exceeds tolerance"

        # Shapes should match
        assert yuv_cpu.shape == yuv_gpu.shape
        assert label_cpu.shape == label_gpu.shape

    def test_train_augmentation_deterministic(self, rgb_chw_single: Tensor, label_single: Tensor) -> None:
        """Test that GPU train augmentation is deterministic with same seed."""
        device = torch.device("cpu")

        # Run GPU version twice with same seed
        torch.manual_seed(456)
        yuv_gpu1, label_gpu1 = train_augment_image_gpu(rgb_chw_single.clone(), label_single.clone(), device)

        torch.manual_seed(456)
        yuv_gpu2, label_gpu2 = train_augment_image_gpu(rgb_chw_single.clone(), label_single.clone(), device)

        # Results should be identical with same seed
        assert torch.allclose(yuv_gpu1, yuv_gpu2, atol=1e-6), "GPU augmentation should be deterministic with same seed"
        assert torch.allclose(label_gpu1, label_gpu2, atol=1e-6), "Labels should be identical with same seed"

    def test_batch_vs_individual_equivalence(self, rgb_chw_batch: Tensor, labels_batch: Tensor) -> None:
        """Test that batch GPU processing produces same results as individual processing."""
        device = torch.device("cpu")
        batch_size = rgb_chw_batch.shape[0]

        # Individual processing
        individual_images = []
        individual_labels = []

        for i in range(batch_size):
            torch.manual_seed(100 + i)  # Different seed per sample
            img, lbl = augment_image_test_mode_gpu(rgb_chw_batch[i], labels_batch[i], device)
            individual_images.append(img)
            individual_labels.append(lbl)

        individual_batch_images = torch.stack(individual_images, dim=0)
        individual_batch_labels = torch.stack(individual_labels, dim=0)

        # Batch processing with same seeds
        torch.manual_seed(100)  # Set base seed
        batch_images, batch_labels = augment_batch_test_mode_gpu(rgb_chw_batch.clone(), labels_batch.clone(), device)

        # For test augmentation (no randomness), results should be identical
        assert torch.allclose(individual_batch_images, batch_images, atol=1e-6), (
            "Batch processing should match individual for test augmentation"
        )
        assert torch.allclose(individual_batch_labels, batch_labels, atol=1e-6), "Batch labels should match individual"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device_processing(self, rgb_chw_single: Tensor, label_single: Tensor) -> None:
        """Test that GPU augmentation actually runs on GPU device."""
        device = torch.device("cuda")

        # Process on GPU
        yuv_gpu, label_gpu = train_augment_image_gpu(rgb_chw_single.clone(), label_single.clone(), device)

        # Results should be on GPU
        assert yuv_gpu.device.type == "cuda", "Output image should be on GPU"
        assert label_gpu.device.type == "cuda", "Output label should be on GPU"

        # Shapes should be preserved
        assert yuv_gpu.shape == rgb_chw_single.shape
        assert label_gpu.shape == label_single.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_numerical_equivalence(self, rgb_chw_single: Tensor, label_single: Tensor) -> None:
        """Test numerical equivalence between CPU and GPU processing."""
        # Process on CPU
        torch.manual_seed(789)
        yuv_cpu, label_cpu = augment_image_test_mode_gpu(
            rgb_chw_single.clone(), label_single.clone(), torch.device("cpu")
        )

        # Process on GPU
        torch.manual_seed(789)
        yuv_gpu, label_gpu = augment_image_test_mode_gpu(
            rgb_chw_single.clone(), label_single.clone(), torch.device("cuda")
        )

        # Move GPU results to CPU for comparison
        yuv_gpu_cpu = yuv_gpu.cpu()
        label_gpu_cpu = label_gpu.cpu()

        # Results should be nearly identical
        max_diff_image = torch.max(torch.abs(yuv_cpu - yuv_gpu_cpu))
        max_diff_label = torch.max(torch.abs(label_cpu - label_gpu_cpu))

        assert max_diff_image < 1e-5, f"CPU/GPU image difference {max_diff_image} exceeds tolerance"
        assert max_diff_label < 1e-5, f"CPU/GPU label difference {max_diff_label} exceeds tolerance"


class TestGPUAugmentationFeatures:
    """Test specific GPU augmentation features and edge cases."""

    def test_device_auto_detection(self) -> None:
        """Test automatic device detection works correctly."""
        rgb = torch.rand(3, 30, 40)
        label = torch.tensor([15.0, 20.0, 5.0])

        # Should use image.device when device=None
        yuv, lbl = augment_image_test_mode_gpu(rgb, label, device=None)
        assert yuv.device == rgb.device
        assert lbl.device == label.device

    def test_device_string_conversion(self) -> None:
        """Test device string conversion works correctly."""
        rgb = torch.rand(3, 30, 40)
        label = torch.tensor([15.0, 20.0, 5.0])

        # Should convert string to device
        yuv, lbl = augment_image_test_mode_gpu(rgb, label, device="cpu")
        assert yuv.device.type == "cpu"
        assert lbl.device.type == "cpu"

    def test_batch_brightness_variation(self) -> None:
        """Test that batch processing applies different brightness to each sample."""
        device = torch.device("cpu")
        batch_size = 4

        # Create identical images
        rgb_batch = torch.ones(batch_size, 3, 30, 40, dtype=torch.float32)
        labels_batch = torch.ones(batch_size, 3) * 10

        # Apply batch augmentation
        torch.manual_seed(999)
        yuv_batch, _ = train_augment_batch_gpu(rgb_batch, labels_batch, device)

        # Check that different samples have different brightness levels
        # (Each sample should have been multiplied by different brightness factor)
        sample_means = torch.mean(yuv_batch, dim=(1, 2, 3))  # Mean per sample

        # Not all samples should be identical (some variation due to brightness)
        std_across_samples = torch.std(sample_means)
        assert std_across_samples > 0.01, "Batch augmentation should create variation across samples"

    def test_flip_probability_distribution(self) -> None:
        """Test that flip probability is approximately 50% over many samples."""
        device = torch.device("cpu")
        num_samples = 1000

        # Create batch of images
        rgb_batch = torch.rand(num_samples, 3, 30, 40, dtype=torch.float32)
        labels_batch = torch.ones(num_samples, 3) * 15  # Center labels

        # Apply batch augmentation
        _, labels_result = train_augment_batch_gpu(rgb_batch, labels_batch, device)

        # Check how many were flipped (flipped samples have different x coordinate)
        original_x = labels_batch[:, 0]
        result_x = labels_result[:, 0]

        # Flipped samples will have x coordinate changed
        from uc_ball_hyp_generator.config import patch_width

        expected_flipped_x = patch_width - original_x

        # Count flips (allowing for small numerical errors)
        flips = torch.abs(result_x - expected_flipped_x) < 0.01
        flip_ratio = flips.float().mean().item()

        # Should be approximately 50% flipped (within reasonable tolerance for random)
        assert 0.4 < flip_ratio < 0.6, f"Flip ratio {flip_ratio:.3f} should be around 0.5"

    def test_empty_batch_handling(self) -> None:
        """Test that empty batches are handled gracefully."""
        device = torch.device("cpu")

        # Create empty batch
        empty_images = torch.empty(0, 3, 30, 40)
        empty_labels = torch.empty(0, 3)

        # Should not crash
        result_images, result_labels = augment_batch_test_mode_gpu(empty_images, empty_labels, device)

        assert result_images.shape == (0, 3, 30, 40)
        assert result_labels.shape == (0, 3)

    def test_single_sample_batch(self) -> None:
        """Test batch processing with single sample."""
        device = torch.device("cpu")

        # Single sample batch
        rgb_batch = torch.rand(1, 3, 30, 40)
        labels_batch = torch.rand(1, 3) * 20

        # Should work correctly
        yuv_batch, labels_result = train_augment_batch_gpu(rgb_batch, labels_batch, device)

        assert yuv_batch.shape == (1, 3, 30, 40)
        assert labels_result.shape == (1, 3)


class TestBallDatasetGPU:
    """Test the BallDatasetGPU class functionality."""

    def test_dataset_initialization(self) -> None:
        """Test BallDatasetGPU initialization."""
        images = ["dummy1.jpg", "dummy2.jpg"]
        labels = [(10, 20, 50, 60), (15, 25, 55, 65)]

        dataset = BallDatasetGPU(images, labels, trainset=True, device="cpu")

        assert len(dataset) == 2
        assert dataset.trainset is True
        assert dataset.device.type == "cpu"

    def test_device_auto_detection_dataset(self) -> None:
        """Test device auto-detection in dataset."""
        images = ["dummy.jpg"]
        labels = [(10, 20, 50, 60)]

        dataset = BallDatasetGPU(images, labels, device=None)

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert dataset.device.type == expected_device


if __name__ == "__main__":
    # Run basic functionality tests
    print("Running GPU augmentation tests...")

    # Test basic GPU functions
    rgb_test = torch.rand(3, 30, 40)
    label_test = torch.tensor([15.0, 20.0, 5.0])

    print("✓ Testing individual GPU augmentation...")
    yuv_result, label_result = test_augment_image_gpu(rgb_test, label_test, torch.device("cpu"))
    print(f"  Individual GPU augmentation works: {rgb_test.shape} → {yuv_result.shape}")

    print("✓ Testing batch GPU augmentation...")
    rgb_batch = torch.rand(4, 3, 30, 40)
    labels_batch = torch.ones(4, 3) * 10
    yuv_batch, labels_batch_result = test_augment_batch_gpu(rgb_batch, labels_batch, torch.device("cpu"))
    print(f"  Batch GPU augmentation works: {rgb_batch.shape} → {yuv_batch.shape}")

    print("✓ Testing dataset GPU initialization...")
    dataset = BallDatasetGPU(["dummy.jpg"], [(10, 20, 50, 60)], device="cpu")
    print(f"  Dataset GPU initialization works: {len(dataset)} samples")

    print("All basic GPU augmentation tests passed! Run with pytest for full test suite.")
