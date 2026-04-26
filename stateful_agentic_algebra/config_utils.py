"""Small YAML/JSON config loader for Stateful Agentic Algebra CLIs.

PyYAML is used when available. A minimal fallback parser supports the simple
top-level scalar/list YAML files shipped in `stateful_agentic_algebra/configs`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML or JSON config file. Empty paths return an empty dict."""

    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = _load_yaml(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return payload


def config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present config value among key aliases."""

    for key in keys:
        if key in config:
            return config[key]
    return default


def csv_default(value: Any) -> str:
    """Convert a scalar/list config value into CLI comma-separated form."""

    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def bool_default(value: Any) -> bool:
    """Coerce common YAML/string bool values."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_key is None:
                raise ValueError("List item found before a key in config")
            result.setdefault(current_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        current_key = key.strip()
        value = raw_value.strip()
        if value == "":
            result[current_key] = []
        elif value.startswith("[") and value.endswith("]"):
            result[current_key] = [_parse_scalar(item.strip()) for item in value[1:-1].split(",") if item.strip()]
        else:
            result[current_key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    unquoted = value.strip().strip('"').strip("'")
    lowered = unquoted.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(unquoted)
    except ValueError:
        pass
    try:
        return float(unquoted)
    except ValueError:
        return unquoted
