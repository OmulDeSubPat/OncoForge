from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipelines.reproducibility import (
    append_metric_history,
    ensure_standard_reproducibility_files,
    export_reproducibility_manifest,
)


class ReproducibilityArtifactsTests(unittest.TestCase):
    def test_bootstrap_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = ensure_standard_reproducibility_files(root)
            manifest = export_reproducibility_manifest(root)

            expected_files = {
                "valori_R2.csv",
                "valori_RMSE.csv",
                "valori_MAE.csv",
                "valori_MSE.csv",
                "valori_pIC50.csv",
                "valori_IC50.csv",
                "valori_Pearson.csv",
                "valori_Spearman.csv",
                "valori_Incertitudine.csv",
                "istoric_metrici.csv",
                "benchmark_studii.csv",
                "comparatii_literatura.csv",
                "legenda_grafice.md",
            }
            present = {path.name for path in created}
            self.assertTrue(expected_files.issubset({Path(item["path"]).name for item in manifest["files"]}))
            self.assertTrue(expected_files.issubset(present))

            for item in manifest["files"]:
                file_path = Path(item["path"])
                self.assertTrue(file_path.exists(), f"Missing reproducibility artifact: {file_path}")
                self.assertGreater(item["size"], 0, f"Artifact should not be empty: {file_path}")

    def test_metric_history_appends_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_metric_history(
                metric_name="R2",
                metric_value=0.81,
                version="v1.0",
                experiment_name="smoke",
                split="scaffold",
                observations="unit-test",
                root=root,
            )

            r2_path = root / "valori_R2.csv"
            history_path = root / "istoric_metrici.csv"

            r2_lines = r2_path.read_text(encoding="utf-8").strip().splitlines()
            history_lines = history_path.read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(len(r2_lines), 2)
            self.assertEqual(len(history_lines), 2)
            self.assertIn("0.81", r2_lines[1])
            self.assertIn("R2", history_lines[1])


if __name__ == "__main__":
    unittest.main()
