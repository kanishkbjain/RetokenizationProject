from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FIGURE_DATA_DIR = DATA_DIR / "figure_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURE_OUTPUT_DIR = OUTPUTS_DIR / "figures"

_RETOK_ROOT_ENV = "RETOK_RETOK_ROOT"
_TYPO_ROOT_ENV = "RETOK_TYPO_ROOT"
_RESULTS_ROOTS_ENV = "RETOK_RESULTS_ROOTS"

DEFAULT_RETOK_ROOT = RAW_DATA_DIR / "Tokenizer_passK"
DEFAULT_TYPO_ROOT = RAW_DATA_DIR / "TokenizationProject"

MODALITY_RESULTS_DIRS = {
    "retok": "results_passretok",
    "typo": "results_passattypos",
}


def _split_paths(value: str) -> list[Path]:
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def get_results_roots() -> tuple[Path, ...]:
    configured = os.environ.get(_RESULTS_ROOTS_ENV)
    if configured:
        return tuple(_split_paths(configured))

    roots = [get_modality_root("retok"), get_modality_root("typo")]
    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        unique_roots.append(root)
    return tuple(unique_roots)


def get_modality_root(modality: str) -> Path:
    if modality == "retok":
        return Path(os.environ.get(_RETOK_ROOT_ENV, DEFAULT_RETOK_ROOT)).expanduser()
    if modality == "typo":
        return Path(os.environ.get(_TYPO_ROOT_ENV, DEFAULT_TYPO_ROOT)).expanduser()
    raise ValueError(f"Unsupported modality: {modality}")


def get_results_dataset_dir(modality: str, dataset: str) -> Path:
    return get_modality_root(modality) / MODALITY_RESULTS_DIRS[modality] / dataset
