from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from uc_ball_hyp_generator.labelingtool.config import load_config
from uc_ball_hyp_generator.labelingtool.gui import LabelingToolWindow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uc-ball-label", description="Qt-based soccer ball labeling tool")
    parser.add_argument("image_path", help="Path to an image file or a directory of images")
    parser.add_argument("--output", help="CSV output path (overrides config)", default=None)
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    parser.add_argument("--stdout", help="Print labels to stdout instead of writing CSV", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    cli_over: dict[str, object] = {}
    if args.stdout:
        cli_over["output"] = "-"
    elif args.output:
        cli_over["output"] = str(args.output)
    if cli_over:
        overrides["cli"] = cli_over

    cfg = load_config(path=args.config, overrides=overrides)

    app = QApplication(sys.argv)
    win = LabelingToolWindow(cfg)
    try:
        win.open_path(Path(args.image_path))
    except Exception as exc:  # noqa: BLE001
        msg: str = str(exc)
        sys.stderr.write(msg + "\n")
        return 2
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
