from pathlib import Path

# Root folder of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# ChEMBL target for human EGFR
EGFR_TARGET_CHEMBL_ID = "CHEMBL203"

# Filters for dataset cleaning
KEEP_STANDARD_TYPE = {"IC50"}
KEEP_STANDARD_UNITS = {"nM"}
KEEP_STANDARD_RELATION = {"="}

# Optional IC50 range (nM)
IC50_NM_MIN = 0.1
IC50_NM_MAX = 100000