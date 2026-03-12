from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.config import PROJECT_ROOT, PROCESSED_DIR
from src.features.featurize_ecfp import ecfp_from_smiles


def featurize_smiles_list(smiles_list):
    return np.vstack([ecfp_from_smiles(s) for s in smiles_list])


def main():
    train_path = PROCESSED_DIR / "egfr_chembl_ic50_clean.csv"
    gen_path = PROJECT_ROOT / "reports" / "iterative_ai_optimized_candidates.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")

    if not gen_path.exists():
        gen_path = PROJECT_ROOT / "reports" / "final_diverse_candidates.csv"

    if not gen_path.exists():
        raise FileNotFoundError("Missing generated candidates file.")

    train_df = pd.read_csv(train_path)
    gen_df = pd.read_csv(gen_path)

    # subeșantionăm train pentru plot mai curat
    train_sample = train_df.sample(min(1500, len(train_df)), random_state=42)
    gen_sample = gen_df.head(min(300, len(gen_df))).copy()

    train_smiles = train_sample["smiles_canonical"].tolist()
    gen_smiles = gen_sample["smiles"].tolist()

    print("[INFO] Featurizing train sample...")
    X_train = featurize_smiles_list(train_smiles)

    print("[INFO] Featurizing generated sample...")
    X_gen = featurize_smiles_list(gen_smiles)

    X_all = np.vstack([X_train, X_gen])

    print("[INFO] Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    n_train = len(X_train)
    train_pca = X_pca[:n_train]
    gen_pca = X_pca[n_train:]

    plt.figure(figsize=(8, 6))
    plt.scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.35, label="Training molecules")
    plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.8, label="Generated candidates")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Chemical Space Projection (PCA of Morgan Fingerprints)")
    plt.legend()
    plt.tight_layout()

    out_path = PROJECT_ROOT / "reports" / "chemical_space_pca.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved chemical space PCA: {out_path}")


if __name__ == "__main__":
    main()