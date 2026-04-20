import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "diffuvqa" / "config.json"


def _normalize_timestep_respacing(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return ""
        try:
            loaded = json.loads(trimmed)
            return loaded
        except json.JSONDecodeError:
            return trimmed
    return value


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_runtime_args(
    checkpoint_path: Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> Namespace:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    defaults = load_json(DEFAULT_CONFIG_PATH)
    training_args_path = checkpoint_path.parent / "training_args.json"
    if training_args_path.exists():
        training_args = load_json(training_args_path)
        defaults.update(training_args)

    defaults["model_path"] = str(checkpoint_path)
    defaults["checkpoint_path"] = str(checkpoint_path.parent)

    if overrides:
        defaults.update({k: v for k, v in overrides.items() if v is not None})

    defaults["timestep_respacing"] = _normalize_timestep_respacing(
        defaults.get("timestep_respacing", "")
    )

    return Namespace(**defaults)
