from __future__ import annotations

import getpass
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from uc_ball_hyp_generator.labelingtool.model import BoundingBox
from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

_logger = get_logger("uc_ball_hyp_generator.labelingtool.persistence")


@dataclass(frozen=True)
class LabelRecord:
    image_file: str
    timestamp_ms: int
    user: str
    class_name: str
    x1: int | None = None
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None
    subclass: str | None = None

    def to_line(self) -> str:
        if self.class_name == "NoBall":
            return f"{self.image_file};{self.timestamp_ms};{self.user};NoBall"
        sx1 = str(int(self.x1 or 0))
        sy1 = str(int(self.y1 or 0))
        sx2 = str(int(self.x2 or 0))
        sy2 = str(int(self.y2 or 0))
        sub = self.subclass or ""
        return f"{self.image_file};{self.timestamp_ms};{self.user};{self.class_name};{sx1};{sy1};{sx2};{sy2};{sub}"

    @property
    def image_key(self) -> str:
        return Path(self.image_file).name


def _now_ms() -> int:
    return int(time.time() * 1000)


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown"


def _parse_line(line: str) -> LabelRecord | None:
    raw = line.strip()
    if not raw:
        return None
    parts = raw.split(";")
    if len(parts) < 4:
        return None
    image_file, ts_str, user, cls = parts[0], parts[1], parts[2], parts[3]
    try:
        ts = int(ts_str)
    except Exception:  # noqa: BLE001
        ts = 0
    if cls == "NoBall":
        return LabelRecord(image_file=image_file, timestamp_ms=ts, user=user, class_name="NoBall")
    if len(parts) < 9:
        return None
    try:
        x1 = int(parts[4])
        y1 = int(parts[5])
        x2 = int(parts[6])
        y2 = int(parts[7])
    except Exception:  # noqa: BLE001
        return None
    subclass = parts[8] if len(parts) >= 9 else None
    return LabelRecord(
        image_file=image_file,
        timestamp_ms=ts,
        user=user,
        class_name=cls,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        subclass=subclass,
    )


def _read_existing(csv_path: Path) -> list[LabelRecord]:
    if not csv_path.is_file():
        return []
    try:
        lines = csv_path.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return []
    out: list[LabelRecord] = []
    for ln in lines:
        if rec := _parse_line(ln):
            out.append(rec)
    return out


def _sort_key(rec: LabelRecord) -> tuple[str, int, str, str]:
    has_sub = 0 if rec.subclass in (None, "") else 1
    sub = rec.subclass or ""
    return (rec.class_name, has_sub, sub, rec.image_key)


def _normalize_entries(
    entries: Iterable[tuple[str | Path, BoundingBox | None]],
) -> list[LabelRecord]:
    user = _current_user()
    ts = _now_ms()
    out: list[LabelRecord] = []
    for img, bb in entries:
        img_name = Path(img).name
        if bb is None:
            out.append(LabelRecord(image_file=img_name, timestamp_ms=ts, user=user, class_name="NoBall"))
            continue
        out.append(
            LabelRecord(
                image_file=img_name,
                timestamp_ms=ts,
                user=user,
                class_name=bb.class_name,
                x1=int(bb.x1),
                y1=int(bb.y1),
                x2=int(bb.x2),
                y2=int(bb.y2),
                subclass=bb.subclass,
            )
        )
    return out


def save_labels(csv_path: str | Path, entries: Iterable[tuple[str | Path, BoundingBox | None]]) -> None:
    """
    Save labels to CSV or stdout following the specified format and sorting.

    If csv_path is "-" or "stdout", print each provided entry as a line to stdout.
    Otherwise, merge with an existing CSV (if present), replace rows for identical
    image filenames, sort, and write back.
    """
    recs = _normalize_entries(entries)
    path_str = str(csv_path)
    if path_str in ("-", "stdout"):
        for r in recs:
            print(r.to_line())
        return

    path = Path(csv_path).expanduser()
    existing = _read_existing(path)
    by_key: dict[str, LabelRecord] = {r.image_key: r for r in existing}
    for r in recs:
        by_key[r.image_key] = r
    merged = sorted(by_key.values(), key=_sort_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(r.to_line() for r in merged) + ("\n" if merged else ""), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _logger.exception("Failed to write CSV %s", path)
        raise exc


def load_existing_labels(csv_path: str | Path) -> dict[str, BoundingBox]:
    """
    Load existing Ball labels from CSV.

    Returns a mapping of base image filename to BoundingBox for rows with a box.
    """
    path = Path(csv_path).expanduser()
    out: dict[str, BoundingBox] = {}
    for r in _read_existing(path):
        if r.class_name == "NoBall":
            continue
        if r.x1 is None or r.y1 is None or r.x2 is None or r.y2 is None:
            continue
        out[r.image_key] = BoundingBox(
            x1=int(r.x1),
            y1=int(r.y1),
            x2=int(r.x2),
            y2=int(r.y2),
            class_name=r.class_name,
            subclass=r.subclass,
        )
    return out


def load_noball_images(csv_path: str | Path) -> set[str]:
    """
    Load image filenames labeled as NoBall from CSV.

    Returns a set of base image filenames.
    """
    path = Path(csv_path).expanduser()
    return {r.image_key for r in _read_existing(path) if r.class_name == "NoBall"}
