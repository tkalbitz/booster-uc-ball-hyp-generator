import statistics
from collections import Counter
from typing import List, Tuple

from uc_ball_hyp_generator.hyp_generator.config import image_dir, testset_csv_collection, trainingset_csv_collection
from uc_ball_hyp_generator.utils.csv_label_reader import load_csv_collection


def _print_histogram(data: List[int], title: str, unit: str = "pixels") -> None:
    """Helper function to print a histogram for given data."""
    if not data:
        return

    min_val = min(data)
    max_val = max(data)
    num_bins = 10
    bin_width = (max_val - min_val) / num_bins if max_val > min_val else 1

    bins = [0] * num_bins
    for value in data:
        if bin_width > 0:
            bin_index = min(int((value - min_val) / bin_width), num_bins - 1)
        else:
            bin_index = 0
        bins[bin_index] += 1

    print(f"\n{title} Histogram:")
    for i in range(num_bins):
        lower_bound = min_val + i * bin_width
        upper_bound = min_val + (i + 1) * bin_width
        count = bins[i]
        print(f"{lower_bound:6.1f} - {upper_bound:6.1f} {unit}: {count}")


def compute_statistics(labels: List[Tuple[int, int, int, int]]) -> None:
    """Compute and print statistics for ball bounding boxes."""
    if not labels:
        print("No labels found to compute statistics.")
        return

    widths: List[int] = []
    heights: List[int] = []
    areas: List[int] = []
    aspect_ratios: List[float] = []

    for x1, y1, x2, y2 in labels:
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height != 0 else 0

        widths.append(width)
        heights.append(height)
        areas.append(area)
        aspect_ratios.append(aspect_ratio)

    # Basic statistics
    avg_width = statistics.mean(widths)
    med_width = statistics.median(widths)
    avg_height = statistics.mean(heights)
    med_height = statistics.median(heights)
    avg_area = statistics.mean(areas)
    med_area = statistics.median(areas)

    print(f"Number of balls: {len(labels)}")
    print(f"Average width: {avg_width:.2f} pixels")
    print(f"Median width: {med_width} pixels")
    print(f"Average height: {avg_height:.2f} pixels")
    print(f"Median height: {med_height} pixels")
    print(f"Average area: {avg_area:.2f} pixels²")
    print(f"Median area: {med_area} pixels²")

    # Histograms
    _print_histogram(widths, "Width")
    _print_histogram(heights, "Height")

    # Most common aspect ratio
    print("\nAspect Ratio Analysis:")
    if aspect_ratios:
        # Round to 2 decimal places for grouping
        rounded_ratios = [round(ratio, 2) for ratio in aspect_ratios]
        ratio_counter = Counter(rounded_ratios)
        most_common_ratio, count = ratio_counter.most_common(1)[0]
        print(f"Most common aspect ratio (width/height): {most_common_ratio} (count: {count})")

        # Print top 5 most common aspect ratios
        print("Top 5 aspect ratios:")
        for ratio, cnt in ratio_counter.most_common(5):
            print(f"  {ratio}: {cnt} balls")


def main():
    png_files: dict[str, str] = {f.name: str(f) for f in image_dir.glob("**/*.png")}

    train_img, train_labels, skipped_trainingset = load_csv_collection(trainingset_csv_collection, png_files)
    test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

    all_labels = train_labels + test_labels
    compute_statistics(all_labels)


if __name__ == "__main__":
    main()
