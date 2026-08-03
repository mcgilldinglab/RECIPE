from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default


DATA_ROOT = _path_from_env("RECIPE_DATA_ROOT", REPO_ROOT / "data")
MODEL_ROOT = _path_from_env("RECIPE_MODEL_ROOT", REPO_ROOT / "models")
OUTPUT_ROOT = _path_from_env("RECIPE_OUTPUT_ROOT", REPO_ROOT / "outputs")
SOURCE_DATA_ROOT = _path_from_env("RECIPE_SOURCE_DATA_ROOT", DATA_ROOT)

BULK_DATA_DIR = DATA_ROOT / "bulk"
NETWORK_DATA_DIR = DATA_ROOT / "networks"
PAUSING_DATA_DIR = DATA_ROOT / "pausing"
SINGLE_CELL_DATA_DIR = DATA_ROOT / "single_cell"

BULK_MODEL_DIR = MODEL_ROOT / "bulk"
PPI_MODEL_DIR = MODEL_ROOT / "ppi"
SINGLE_CELL_MODEL_DIR = MODEL_ROOT / "single_cell"


def ensure_output_dir(*parts: str) -> Path:
    path = OUTPUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
