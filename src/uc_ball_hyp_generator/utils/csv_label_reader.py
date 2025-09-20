import re
from pathlib import Path

from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


def read_csv_label(
    csv_file: Path, img_files: dict[str, str]
) -> tuple[tuple[list[str], list[tuple[int, int, int, int]], list[str], list[str]], int]:
    ball_regex = re.compile(r"(.+_U\.png);\d+;[^;]+;Ball;(\d+);(\d+);(\d+);(\d+);Ball")
    noball_regex = re.compile(r"(.+_U\.png);[^;]+;[^;]+;NoBall")
    ignore_regex = re.compile(r"(.+_U\.png);\d+;[^;]+;Ignore")

    skipped_balls: int = 0
    positive_result: dict[str, tuple[str, tuple[int, int, int, int], str]] = {}
    negative_result: dict[str, str] = {}
    ignored_files: set[str] = set()
    processed_files: set[str] = set()

    for line in csv_file.open():
        line = line.strip()
        if not line:
            continue

        # Handle Ignore directive first - removes from both positive and negative
        m = ignore_regex.match(line)
        if m:
            img_file = m.group(1)
            ignored_files.add(img_file)
            positive_result.pop(img_file, None)
            negative_result.pop(img_file, None)
            continue

        # Handle Ball labels (positive samples)
        m = ball_regex.match(line)
        if m:
            img_file = m.group(1)

            if img_file in ignored_files:
                continue

            x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))

            if x1 > x2:
                x1, x2 = x2, x1

            if y1 > y2:
                y1, y2 = y2, y1

            if img_file not in img_files:
                _logger.warning("Skip missing image file %s", img_file)
                continue

            if img_file in processed_files:
                skipped_balls += 1
                positive_result.pop(img_file, None)
                negative_result.pop(img_file, None)
                continue

            processed_files.add(img_file)

            # We only interested with labels with a certain size
            if x2 - x1 < 4 or y2 - y1 < 4:
                skipped_balls += 1
                continue

            positive_result[img_file] = (img_files[img_file], (x1, y1, x2, y2), line)
            negative_result.pop(img_file, None)  # Remove from negative if it was there
            continue

        # Handle NoBall labels (negative samples)
        m = noball_regex.match(line)
        if m:
            img_file = m.group(1)

            if img_file in ignored_files or img_file in processed_files:
                continue

            if img_file not in img_files:
                _logger.warning("Skip missing image file %s", img_file)
                continue

            processed_files.add(img_file)
            negative_result[img_file] = img_files[img_file]

    # Prepare return values
    if len(positive_result) == 0:
        positive_imgs: list[str] = []
        positive_labels: list[tuple[int, int, int, int]] = []
        positive_lines: list[str] = []
    else:
        values = list(positive_result.values())
        pos_imgs_tuple, pos_labels_tuple, pos_lines_tuple = zip(*values, strict=False)
        positive_imgs = list(pos_imgs_tuple)
        positive_labels = list(pos_labels_tuple)
        positive_lines = list(pos_lines_tuple)

    negative_imgs = list(negative_result.values())

    return (positive_imgs, positive_labels, positive_lines, negative_imgs), skipped_balls


def load_csv_collection(
    file: Path, img_files: dict[str, str]
) -> tuple[list[str], list[tuple[int, int, int, int]], list[str], int]:
    res_positive_imgs: list[str] = []
    res_labels: list[tuple[int, int, int, int]] = []
    res_lines: list[str] = []
    res_negative_imgs: list[str] = []
    skipped: int = 0

    for csv_file in file.open():
        [positive_imgs, labels, lines, negative_imgs], ignored = read_csv_label(
            file.parent / csv_file.strip(), img_files
        )
        res_positive_imgs += positive_imgs
        res_labels += labels
        res_lines += lines
        res_negative_imgs += negative_imgs
        skipped += ignored

    return res_positive_imgs, res_labels, res_negative_imgs, skipped


if __name__ == "__main__":
    start_dir = Path("/home/tkalbitz/temp/BallImages/")

    cnt_balls: int = 0
    cnt_no_balls: int = 0
    to_small: int = 0

    png_files: dict[str, str] = {f.name: str(f) for f in start_dir.glob("**/*.png")}

    for csv_file in list(start_dir.glob("**/labels.csv")):
        [positive_imgs, ball_labels, _, negative_imgs], file_to_small = read_csv_label(csv_file, png_files)
        cnt_balls += len(positive_imgs)
        cnt_no_balls += len(negative_imgs)
        to_small += file_to_small

    _logger.info("Too small balls: %d", to_small)
    _logger.info("Found valid balls: %d", cnt_balls)
    _logger.info("Found no-ball images: %d", cnt_no_balls)
