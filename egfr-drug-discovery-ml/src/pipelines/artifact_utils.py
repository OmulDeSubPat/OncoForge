from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv_artifact(path: Path, required_columns: list[str] | None = None, producer: str | None = None) -> pd.DataFrame:
    if not path.exists():
        message = f"Missing artifact: {path}"
        if producer:
            message += f"\nRun: {producer}"
        raise FileNotFoundError(message)

    df = pd.read_csv(path, low_memory=False)

    if required_columns:
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            message = f"Artifact {path} is missing columns: {missing}"
            if producer:
                message += f"\nRegenerate with: {producer}"
            raise ValueError(message)

    return df
