"""Configuration loading. All parameters live in configs/*.json (SI units)."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_new_dir(path, overwrite=False):
    """Create an output directory, refusing to silently overwrite earlier results."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists and is not empty. Use a new --tag/--name, "
                f"or pass --overwrite to replace it explicitly.")
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
