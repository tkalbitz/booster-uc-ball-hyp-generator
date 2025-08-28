import os
import re
import shutil
import sys
import collections
from pathlib import Path


def read_csv_label(csv_file: Path, img_files: dict) -> list:
    regex = re.compile(r"(.+_U\.png);\d+;[^;]+;Ball;(\d+);(\d+);(\d+);(\d+);Ball")
    ignore = re.compile(r"(.+_U\.png);\d+;[^;]+;Ignore");

    skipped_balls = 0
    result = {}
    found_files = set()

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
            print(f"Skip missing image file {img_file}")
            continue

        if img_file in found_files:
            skipped_balls += 1
            result.pop(img_file, None)
            # print(f"Skip file with more than one ball {img_file}")
            continue

        found_files.add(img_file)

        # We only interested with labels with a certain size
        if x2 - x1 < 4 or y2 - y1 < 4:
            skipped_balls += 1
            continue

        result[img_file] = (img_files[img_file], (x1, y1, x2, y2), line)

    if len(result.values()) == 0:
        return [[], [], []], skipped_balls

    return list(zip(*result.values())), skipped_balls


def load_csv_collection(file: Path, img_files: dict):
    res_imgs = []
    res_labels = []
    res_lines = []
    skipped = 0

    for csv_file in file.open():
        [imgs, labels, lines], ignored = read_csv_label(file.parent / csv_file.strip(), img_files)
        res_imgs += imgs
        res_labels += labels
        res_lines += lines
        skipped += ignored

    # print([(item, count) for item, count in collections.Counter(res_imgs).items() if count > 1])
    #os.makedirs('/home/tkalbitz/temp/u_ball_bench/')
    #for f in res_imgs:
    #    shutil.copy2(f, "/home/tkalbitz/temp/u_ball_bench/")
    #with open("/home/tkalbitz/temp/u_ball_bench/labels.csv", "w") as f:
    #    for l in res_lines:
    #        f.write(l)
    #print("XXXXXXXXXXXXXXXXXXXXXXXX")

    return res_imgs, res_labels, skipped


if __name__ == '__main__':
    start_dir = Path('/home/tkalbitz/temp/BallImages/')

    cnt_balls = 0
    to_small = 0

    png_files = {f.name: f for f in start_dir.glob("**/*.png")}

    for csv_file in list(start_dir.glob("**/labels.csv")):
        [img_path, ball_label, _], file_to_small = read_csv_label(csv_file, png_files)
        cnt_balls += len(img_path)
        to_small += file_to_small

    print(f"To small balls: {to_small}")
    print(f"Found valid balls {cnt_balls}")
