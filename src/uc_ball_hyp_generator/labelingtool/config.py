from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict, cast

import yaml  # type: ignore

from uc_ball_hyp_generator.labelingtool.model import Shape
from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

_logger = get_logger("uc_ball_hyp_generator.labelingtool.config")

_SUPPORTED_SHAPES = {s.value for s in Shape}


class SamConfig(TypedDict):
    model_name: str
    cache_dir: str


class CliConfig(TypedDict):
    output: str | None
    log_level: str


class LabelingConfig(TypedDict):
    sam: SamConfig
    cli: CliConfig
    shape: dict[str, str]


def _default_config() -> LabelingConfig:
    """Return the built-in defaults."""
    return {
        "sam": {
            "model_name": "sam_vit_h_4b",
            "cache_dir": "~/.cache/uc_ball_hyp_generator/models/",
        },
        "cli": {
            "output": None,  # defaults to <image_dir>/labels.csv
            "log_level": "INFO",
        },
        "shape": {
            # Default shape per class
            "Ball": "ellipse",
            "NoBall": "rectangle",
        },
    }


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """
    Deep-merge two dictionaries. 'override' wins on conflicts.

    Lists and scalars are replaced; nested dicts are merged recursively.
    """
    out = dict(base)
    for k, v in override.items():
        existing = out.get(k)
        if isinstance(v, dict) and isinstance(existing, dict):
            out[k] = _deep_merge(cast(dict[str, object], existing), cast(dict[str, object], v))
        else:
            out[k] = v
    return out


def _resolve_config_path(path: str | Path | None) -> Path | None:
    """
    Determine the config file path using the search order defined in requirements.md:
      1. Path supplied via --config (if provided here as 'path').
      2. Current working directory: ./config.yaml
      3. ~/.config/uc_ball_hyp_generator/labelingtool/config.yaml
    """
    if path is not None:
        p = Path(path).expanduser()
        return p if p.is_file() else None

    cwd_path = Path.cwd() / "config.yaml"
    if cwd_path.is_file():
        return cwd_path

    home_path = Path("~/.config/uc_ball_hyp_generator/labelingtool/config.yaml").expanduser()
    if home_path.is_file():
        return home_path

    return None


def _load_yaml_file(file_path: Path) -> dict[str, object]:
    """Load a YAML file and return a top-level mapping; raise RuntimeError on I/O or parse errors."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"Config file not found: {file_path}"
        raise RuntimeError(msg) from exc
    except OSError as exc:
        msg = f"Failed to read config file: {file_path}"
        raise RuntimeError(msg) from exc

    try:
        data = yaml.safe_load(text) or {}
    except Exception as exc:  # type: ignore[attr-defined]  # noqa: BLE001
        try:
            _ = yaml.YAMLError  # type: ignore[attr-defined]
            is_yaml_err = isinstance(exc, yaml.YAMLError)  # type: ignore[attr-defined]
        except Exception:
            is_yaml_err = False
        if is_yaml_err:
            msg = f"YAML parse error in config: {file_path}"
            raise RuntimeError(msg) from exc
        msg = f"Unexpected error while parsing YAML: {file_path}"
        raise RuntimeError(msg) from exc

    if not isinstance(data, dict):
        msg = f"Config file must contain a mapping at the top level: {file_path}"
        raise RuntimeError(msg)
    return data


def _validate_shapes(cfg: dict[str, object]) -> None:
    """Validate the 'shape' mapping and raise RuntimeError on invalid entries."""
    shape_map = cfg.get("shape", {})
    if not isinstance(shape_map, dict):
        msg = "Config 'shape' section must be a mapping of {class_name: shape}."
        raise RuntimeError(msg)

    invalid: list[str] = []
    for cls_name, shape_value in shape_map.items():
        if not isinstance(shape_value, str) or shape_value.strip().lower() not in _SUPPORTED_SHAPES:
            invalid.append(f"{cls_name}={shape_value!r}")

    if invalid:
        joined = ", ".join(invalid)
        msg = "Invalid shape values in config: " + joined + ". Supported: rectangle, circle, ellipse."
        raise RuntimeError(msg)


def _expand_paths(cfg: dict[str, object]) -> dict[str, object]:
    """Expand user (~) and env vars in path-like config values."""
    out = dict(cfg)
    sam_obj = out.get("sam", {})
    sam_section = cast(dict[str, object], sam_obj) if isinstance(sam_obj, dict) else {}
    cache_dir = sam_section.get("cache_dir")
    if isinstance(cache_dir, str):
        sam_section["cache_dir"] = os.path.expandvars(os.path.expanduser(cache_dir))
    out["sam"] = sam_section
    return out


# Cache of the last loaded configuration for helpers like get_shape_for_class().
_last_loaded_config: dict[str, object] | None = None


def load_config(path: str | Path | None = None, overrides: dict[str, object] | None = None) -> dict[str, object]:
    """
    Load the labeling tool configuration.

    Merges:
      - Built-in defaults
      - User YAML config (if found by search order or explicit path)
      - Optional overrides (highest precedence)

    Args:
        path: Optional explicit path to a YAML config file.
        overrides: Optional dictionary of values overriding both defaults and file values.

    Returns:
        A merged configuration dictionary with keys: 'sam', 'cli', 'shape'.
    """
    defaults = _default_config()
    file_path = _resolve_config_path(path)

    merged: dict[str, object] = dict(defaults)
    if file_path is not None:
        user = _load_yaml_file(file_path)
        merged = _deep_merge(merged, user)

    if overrides:
        merged = _deep_merge(merged, overrides)

    merged = _expand_paths(merged)
    _validate_shapes(merged)

    global _last_loaded_config
    _last_loaded_config = merged
    _logger.debug("Configuration loaded. Using file: %s", file_path)
    return merged


def get_shape_for_class(class_name: str) -> Shape:
    """
    Resolve the configured Shape for a given class name.

    If no config has been loaded yet, defaults are used.

    Args:
        class_name: The class name whose shape should be returned.

    Returns:
        Shape: The shape configured for the class, defaulting to RECTANGLE if unknown.
    """
    if _last_loaded_config is not None:
        cfg_map: dict[str, object] = _last_loaded_config
    else:
        cfg_map = cast(dict[str, object], _default_config())
    shape_map_obj = cfg_map.get("shape", {})
    shape_map = cast(dict[str, object], shape_map_obj) if isinstance(shape_map_obj, dict) else {}
    shape_str = str(shape_map.get(class_name, "rectangle"))
    return Shape.from_string(shape_str)
