from __future__ import annotations

from pathlib import Path

from uc_ball_hyp_generator.labelingtool.model import BoundingBox
from uc_ball_hyp_generator.labelingtool.persistence import (
    load_existing_labels,
    load_noball_images,
    save_labels,
)


def test_save_and_load_roundtrip(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "labels.csv"
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"

    bb1 = BoundingBox(x1=10, y1=20, x2=30, y2=40, class_name="Ball", subclass="Ball")

    save_labels(csv_path, [(img1, bb1), (img2, None)])

    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("img1.png" in ln and ";Ball;" in ln for ln in content)
    assert any(ln.endswith(";NoBall") and "img2.png" in ln for ln in content)

    loaded = load_existing_labels(csv_path)
    assert "img1.png" in loaded
    assert loaded["img1.png"].as_tuple() == (10, 20, 30, 40)

    noballs = load_noball_images(csv_path)
    assert "img2.png" in noballs


def test_stdout_mode(tmp_path: Path, capsys) -> None:
    img = tmp_path / "img.png"
    bb = BoundingBox(x1=1, y1=2, x2=3, y2=4, class_name="Ball", subclass="Ball")
    save_labels("-", [(img, bb)])
    out = capsys.readouterr().out.strip()
    assert "img.png" in out and ";Ball;" in out
