from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.agents.multi_agent import build_default_scorer, score_smiles_list
from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data.dataset_registry import resolve_preferred_processed_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score and rank the processed EGFR dataset.")
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Score only a subset of molecules for a quick audit run.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="How to select molecules when --max-molecules is provided.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"),
        help="Output CSV path for the ranked artifact.",
    )
    args = parser.parse_args(argv)

    data_path = resolve_preferred_processed_dataset()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {data_path}\n"
            "Run: python -m src.data.fetch_chembl_egfr && python -m src.data.clean_egfr_ic50"
        )

    df = pd.read_csv(data_path)

    if "smiles_canonical" not in df.columns:
        raise ValueError(
            f"Expected column 'smiles_canonical'. Found: {list(df.columns)}"
        )

    smiles_series = df["smiles_canonical"].dropna()
    if args.max_molecules is not None:
        sample_size = min(args.max_molecules, len(smiles_series))
        if args.sample_mode == "random":
            smiles_series = smiles_series.sample(sample_size, random_state=42)
        else:
            smiles_series = smiles_series.head(sample_size)

    smiles_list = smiles_series.tolist()
    scorer = build_default_scorer()
    out = score_smiles_list(smiles_list, scorer=scorer)

    if out.empty:
        raise ValueError("No molecules were successfully scored.")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n[OK] Saved ranked dataset: {out_path}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
