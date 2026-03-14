from __future__ import annotations

from typing import Iterable, Sequence

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str):
        return None
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


def morgan_fp(
    smiles: str | None = None,
    mol=None,
    radius: int = 2,
    n_bits: int = 2048,
):
    if mol is None:
        mol = mol_from_smiles(smiles or "")
    if mol is None:
        return None
    if radius == 2 and n_bits == 2048:
        return _MORGAN_GENERATOR.GetFingerprint(mol)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprint(mol)


def tanimoto_similarity(fp1, fp2) -> float:
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def bulk_tanimoto_similarity(fp, ref_fps: Sequence) -> list[float]:
    if fp is None or not ref_fps:
        return []
    return [float(x) for x in DataStructs.BulkTanimotoSimilarity(fp, list(ref_fps))]


def top_k_mean(values: Iterable[float], k: int = 5) -> float:
    ordered = sorted((float(v) for v in values), reverse=True)
    if not ordered:
        return 0.0
    subset = ordered[: max(1, k)]
    return float(sum(subset) / len(subset))


def murcko_scaffold_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True)
