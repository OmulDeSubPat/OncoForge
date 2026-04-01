from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT
from src.pipelines.build_competition_report_assets import build_assets as build_competition_report_assets
from src.visualization.technical_notebook_plots import NOTEBOOK_DIR, build_assets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build technical notebook plots and markdown assets.")
    parser.add_argument(
        "--ranked-path",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "ranked_egfr_dataset.csv"),
        help="Path to the ranked dataset artifact.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(NOTEBOOK_DIR),
        help="Directory where notebook assets should be written.",
    )
    args = parser.parse_args(argv)

    ranked_path = Path(args.ranked_path)
    out_dir = Path(args.out_dir)

    if not ranked_path.is_absolute():
        ranked_path = PROJECT_ROOT / ranked_path
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    build_assets(ranked_path=ranked_path, out_dir=out_dir)
    build_competition_report_assets(out_dir=out_dir)


if __name__ == "__main__":
    main()
