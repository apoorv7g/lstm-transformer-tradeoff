"""Repo-root paths and YAML config loading.

Every path in configs/*.yaml is relative to the repo root, so scripts behave the same no matter
which directory they are launched from.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name):
    """Load configs/{name}.yaml as a dict, e.g. load_config("data")."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def repo_path(relative):
    """Resolve a config path (relative to the repo root) to an absolute Path."""
    path = Path(relative)
    return path if path.is_absolute() else REPO_ROOT / path
