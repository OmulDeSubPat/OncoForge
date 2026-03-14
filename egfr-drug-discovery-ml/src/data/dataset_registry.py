from __future__ import annotations

from pathlib import Path

from src.config import PROCESSED_DIR


DEFAULT_DATASET_CANDIDATES = [
    PROCESSED_DIR / "egfr_multisource_ic50_clean.csv",
    PROCESSED_DIR / "egfr_chembl_ic50_clean.csv",
]


def resolve_preferred_processed_dataset() -> Path:
    for path in DEFAULT_DATASET_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_DATASET_CANDIDATES[-1]


def dataset_label_from_path(path: Path) -> str:
    name = path.name.lower()
    if "multisource" in name:
        return "multisource"
    if "bindingdb" in name:
        return "bindingdb"
    if "chembl" in name:
        return "chembl"
    return path.stem
