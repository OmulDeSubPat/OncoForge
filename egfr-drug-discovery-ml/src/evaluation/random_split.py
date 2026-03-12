from __future__ import annotations

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)