import re
from pathlib import Path

from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


def read_csv_label(
    csv_file: Path, img_files: dict[str, str]
) -> tuple[tuple[list[str], list[tuple[int, int, int, int]], list[str]], int]:
    regex = re.compile(r"(.+_U\.png);\d+;[^;]+;Ball;(\d+);(\d+);(\d+);(\d+);Ball")
    ignore = re.compile(r"(.+_U\.png);\d+;[^;]+;Ignore")

    skipped_balls: int = 0
    result: dict[str, tuple[str, tuple[int, int, int, int], str]] = {}
    found_files: set[str] = set()

    for line in csv_file.open():
        # Complete file should be ignored
        m = ignore.match(line)
        if m:
            img_file = m.group(1)
            found_files.add(img_file)
            result.pop(img_file, None)
            continue

        m = regex.match(line)

        if not m:
            continue

        img_file = m.group(1)

        x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))

        if x1 > x2:
            x1, x2 = x2, x1

        if y1 > y2:
            y1, y2 = y2, y1

        if img_file not in img_files:
            _logger.warning("Skip missing image file %s", img_file)
            continue

        if img_file in found_files:
            skipped_balls += 1
            result.pop(img_file, None)
            continue

        found_files.add(img_file)

        # We only interested with labels with a certain size
        if x2 - x1 < 4 or y2 - y1 < 4:
            skipped_balls += 1
            continue

        result[img_file] = (img_files[img_file], (x1, y1, x2, y2), line)

    if len(result.values()) == 0:
        empty_imgs: list[str] = []
        empty_labels: list[tuple[int, int, int, int]] = []
        empty_lines: list[str] = []
        return (empty_imgs, empty_labels, empty_lines), skipped_balls

    values = list(result.values())
    imgs, labels, lines = zip(*values)
    return (list(imgs), list(labels), list(lines)), skipped_balls


def load_csv_collection(
    file: Path, img_files: dict[str, str]
) -> tuple[list[str], list[tuple[int, int, int, int]], int]:
    res_imgs: list[str] = []
    res_labels: list[tuple[int, int, int, int]] = []
    res_lines: list[str] = []
    skipped: int = 0

    for csv_file in file.open():
        [imgs, labels, lines], ignored = read_csv_label(file.parent / csv_file.strip(), img_files)
        res_imgs += imgs
        res_labels += labels
        res_lines += lines
        skipped += ignored

    return res_imgs, res_labels, skipped


if __name__ == "__main__":
    start_dir = Path("/home/tkalbitz/temp/BallImages/")

    cnt_balls: int = 0
    to_small: int = 0

    png_files: dict[str, str] = {f.name: str(f) for f in start_dir.glob("**/*.png")}

    for csv_file in list(start_dir.glob("**/labels.csv")):
        [img_path, ball_label, _], file_to_small = read_csv_label(csv_file, png_files)
        cnt_balls += len(img_path)
        to_small += file_to_small

    _logger.info("Too small balls: %d", to_small)
    _logger.info("Found valid balls: %d", cnt_balls)
