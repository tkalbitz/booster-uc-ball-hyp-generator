from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure 'src' is importable without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uc_ball_hyp_generator.labelingtool import get_shape_for_class, load_config  # noqa: E402
from uc_ball_hyp_generator.labelingtool.model import Shape  # noqa: E402


def write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_config_defaults_and_merge(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    write_yaml(
        cfg_file,
        """
sam:
  model_name: "custom_sam"
cli:
  log_level: "DEBUG"
shape:
  Ball: ellipse
  NoBall: rectangle
  FancyBall: circle
""",
    )

    cfg = load_config(cfg_file)
    assert cfg["sam"]["model_name"] == "custom_sam"
    assert cfg["cli"]["log_level"] == "DEBUG"

    # Defaults still present
    assert "cache_dir" in cfg["sam"]
    assert isinstance(cfg["shape"], dict)

    # Custom class merged
    assert cfg["shape"]["FancyBall"] == "circle"


def test_invalid_shape_value_raises(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    write_yaml(
        cfg_file,
        """
shape:
  Ball: hexagon
""",
    )

    with pytest.raises(RuntimeError):
        _ = load_config(cfg_file)


def test_get_shape_for_class_uses_loaded_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    write_yaml(
        cfg_file,
        """
shape:
  Ball: ellipse
  NoBall: rectangle
  Custom: circle
""",
    )

    _ = load_config(cfg_file)
    assert get_shape_for_class("Ball") is Shape.ELLIPSE
    assert get_shape_for_class("Custom") is Shape.CIRCLE
    # Unknown classes default to RECTANGLE
    assert get_shape_for_class("Unknown") is Shape.RECTANGLE
