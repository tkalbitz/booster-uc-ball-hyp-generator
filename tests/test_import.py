from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'src' is importable without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_package_import_and_symbols() -> None:
    import uc_ball_hyp_generator.labelingtool as lt

    assert hasattr(lt, "BoundingBox")
    assert hasattr(lt, "load_config")
    assert hasattr(lt, "get_shape_for_class")
    assert hasattr(lt, "label_image")
    assert hasattr(lt, "save_labels")

    cfg = lt.load_config(None)
    assert isinstance(cfg, dict)
    for key in ("sam", "cli", "shape"):
        assert key in cfg
