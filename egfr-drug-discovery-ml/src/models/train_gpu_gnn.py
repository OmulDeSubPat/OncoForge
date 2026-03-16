from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

try:
    import torch_directml
except ImportError:  # pragma: no cover - optional backend
    torch_directml = None

from src.config import PROJECT_ROOT
from src.data.dataset_registry import dataset_label_from_path, resolve_preferred_processed_dataset
from src.evaluation.random_split import random_split
from src.evaluation.scaffold_split import scaffold_split
from src.evaluation.temporal_split import temporal_split
from src.features.descriptor_features import DESCRIPTOR_NAMES, descriptor_vector_from_smiles
from src.models.train_multiview_ensemble import build_features, evaluate_ensemble, make_model_bundles


COMMON_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATION_ORDER = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


def _resolve_device(prefer_gpu: bool = True) -> tuple[torch.device, str]:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if prefer_gpu and torch_directml is not None:
        try:
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def _metrics_dict(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }


def _atom_features(atom: Chem.Atom) -> np.ndarray:
    atomic_num = atom.GetAtomicNum()
    atom_one_hot = [1.0 if atomic_num == candidate else 0.0 for candidate in COMMON_ATOMS]
    atom_one_hot.append(1.0 if atomic_num not in COMMON_ATOMS else 0.0)
    hybridization = atom.GetHybridization()
    hybrid_one_hot = [1.0 if hybridization == candidate else 0.0 for candidate in HYBRIDIZATION_ORDER]
    return np.asarray(
        atom_one_hot
        + hybrid_one_hot
        + [
            min(atom.GetTotalDegree(), 5) / 5.0,
            min(atom.GetTotalNumHs(), 4) / 4.0,
            min(abs(atom.GetFormalCharge()), 3) / 3.0,
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
            atom.GetMass() / 200.0,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class GraphRecord:
    row_id: int
    smiles: str
    node_features: np.ndarray
    edge_index: np.ndarray
    descriptors: np.ndarray
    target: float


def _smiles_to_graph_record(row_id: int, smiles: str, target: float) -> GraphRecord:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for graph conversion: {smiles}")
    node_features = np.vstack([_atom_features(atom) for atom in mol.GetAtoms()]).astype(np.float32)
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.append((begin, end))
        edges.append((end, begin))
    if not edges:
        edges.append((0, 0))
    return GraphRecord(
        row_id=int(row_id),
        smiles=str(smiles),
        node_features=node_features,
        edge_index=np.asarray(edges, dtype=np.int64),
        descriptors=descriptor_vector_from_smiles(smiles).astype(np.float32),
        target=float(target),
    )


class GraphRecordDataset(Dataset):
    def __init__(self, records: list[GraphRecord], target_mean: float, target_std: float):
        self.records = records
        self.target_mean = float(target_mean)
        self.target_std = float(max(1e-6, target_std))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "row_id": record.row_id,
            "smiles": record.smiles,
            "node_features": record.node_features,
            "edge_index": record.edge_index,
            "descriptors": record.descriptors,
            "target": record.target,
            "target_scaled": (record.target - self.target_mean) / self.target_std,
        }


def _collate_graph_batch(batch: list[dict]) -> dict:
    max_nodes = max(item["node_features"].shape[0] for item in batch)
    node_dim = batch[0]["node_features"].shape[1]
    desc_dim = batch[0]["descriptors"].shape[0]
    batch_size = len(batch)
    x = torch.zeros((batch_size, max_nodes, node_dim), dtype=torch.float32)
    adj = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    descriptors = torch.zeros((batch_size, desc_dim), dtype=torch.float32)
    target = torch.zeros((batch_size,), dtype=torch.float32)
    target_scaled = torch.zeros((batch_size,), dtype=torch.float32)
    row_ids = np.zeros((batch_size,), dtype=np.int64)
    smiles = []
    for idx, item in enumerate(batch):
        node_features = torch.from_numpy(item["node_features"])
        n_nodes = node_features.shape[0]
        x[idx, :n_nodes] = node_features
        mask[idx, :n_nodes] = 1.0
        adj[idx, :n_nodes, :n_nodes] = torch.eye(n_nodes, dtype=torch.float32)
        for src, dst in item["edge_index"]:
            if src < n_nodes and dst < n_nodes:
                adj[idx, src, dst] = 1.0
        degrees = adj[idx, :n_nodes, :n_nodes].sum(dim=-1).clamp(min=1.0)
        norm = torch.diag(torch.pow(degrees, -0.5))
        adj[idx, :n_nodes, :n_nodes] = norm @ adj[idx, :n_nodes, :n_nodes] @ norm
        descriptors[idx] = torch.from_numpy(item["descriptors"])
        target[idx] = float(item["target"])
        target_scaled[idx] = float(item["target_scaled"])
        row_ids[idx] = int(item["row_id"])
        smiles.append(str(item["smiles"]))
    return {
        "row_id": row_ids,
        "smiles": smiles,
        "x": x,
        "adj": adj,
        "mask": mask,
        "descriptors": descriptors,
        "target": target,
        "target_scaled": target_scaled,
    }


class GraphConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(adj, x)
        h = self.linear(h)
        h = self.norm(h)
        h = F.gelu(h)
        return self.dropout(h)


class GraphDescriptorRegressor(nn.Module):
    def __init__(self, node_dim: int, descriptor_dim: int, hidden_dim: int = 96, dropout: float = 0.10):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GraphConvBlock(node_dim, hidden_dim, dropout),
                GraphConvBlock(hidden_dim, hidden_dim, dropout),
                GraphConvBlock(hidden_dim, hidden_dim, dropout),
            ]
        )
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, adj)
        mask_expanded = mask.unsqueeze(-1)
        mean_pool = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        max_pool = h.masked_fill(mask_expanded == 0, -1e9).amax(dim=1)
        descriptor_emb = self.descriptor_mlp(descriptors)
        return self.head(torch.cat([mean_pool, max_pool, descriptor_emb], dim=-1)).squeeze(-1)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def _predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, target_mean: float, target_std: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    row_ids: list[int] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, device)
            pred_scaled = model(moved["x"], moved["adj"], moved["mask"], moved["descriptors"])
            pred = pred_scaled.detach().cpu().numpy() * target_std + target_mean
            y_pred.extend(pred.tolist())
            y_true.extend(batch["target"].numpy().tolist())
            row_ids.extend(batch["row_id"].tolist())
    return np.asarray(row_ids, dtype=np.int64), np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def _fit_gnn_split(
    records_by_row: dict[int, GraphRecord],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    early_patience: int,
) -> dict[str, object]:
    rng = np.random.default_rng(42)
    train_idx = np.asarray(train_idx, dtype=int)
    shuffled_train = rng.permutation(train_idx)
    val_size = max(1, int(0.10 * len(shuffled_train)))
    val_idx = shuffled_train[:val_size]
    fit_idx = shuffled_train[val_size:]
    if len(fit_idx) == 0:
        fit_idx = val_idx
    fit_records = [records_by_row[int(idx)] for idx in fit_idx]
    val_records = [records_by_row[int(idx)] for idx in val_idx]
    test_records = [records_by_row[int(idx)] for idx in np.asarray(test_idx, dtype=int)]
    y_fit = np.asarray([record.target for record in fit_records], dtype=float)
    target_mean = float(y_fit.mean())
    target_std = float(max(y_fit.std(ddof=0), 1e-6))
    train_loader = DataLoader(GraphRecordDataset(fit_records, target_mean, target_std), batch_size=batch_size, shuffle=True, collate_fn=_collate_graph_batch)
    val_loader = DataLoader(GraphRecordDataset(val_records, target_mean, target_std), batch_size=batch_size, shuffle=False, collate_fn=_collate_graph_batch)
    test_loader = DataLoader(GraphRecordDataset(test_records, target_mean, target_std), batch_size=batch_size, shuffle=False, collate_fn=_collate_graph_batch)
    sample = fit_records[0]
    model = GraphDescriptorRegressor(sample.node_features.shape[1], sample.descriptors.shape[0], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    stalled = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            moved = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_scaled = model(moved["x"], moved["adj"], moved["mask"], moved["descriptors"])
            loss = F.smooth_l1_loss(pred_scaled, moved["target_scaled"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        _, y_val, pred_val = _predict_loader(model, val_loader, device, target_mean, target_std)
        val_metrics = _metrics_dict(y_val, pred_val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, "val_rmse": val_metrics["rmse"], "val_r2": val_metrics["r2"]})
        if val_metrics["rmse"] + 1e-5 < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= early_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_row_ids, y_test, pred_test = _predict_loader(model, test_loader, device, target_mean, target_std)
    return {
        "metrics": _metrics_dict(y_test, pred_test),
        "predictions": pd.DataFrame({"row_id": test_row_ids, "y_true": y_test, "pred_gpu_gnn": pred_test}),
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
    }


def _evaluate_reference_ensemble(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    feature_store = build_features(df, "smiles_canonical")
    y = df["pIC50_median"].to_numpy(dtype=float)
    metrics = evaluate_ensemble(make_model_bundles(), feature_store, y, train_idx, test_idx)
    pred_df = pd.DataFrame({"row_id": np.asarray(test_idx, dtype=int), "y_true": y[np.asarray(test_idx, dtype=int)], "pred_reference_ensemble": np.asarray(metrics["pred_mean"], dtype=float)})
    return _metrics_dict(pred_df["y_true"].to_numpy(dtype=float), pred_df["pred_reference_ensemble"].to_numpy(dtype=float)), pred_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a GPU graph regressor and benchmark it against the classical multiview ensemble.")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-patience", type=int, default=6)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args(argv)

    device, device_label = _resolve_device(prefer_gpu=not args.cpu_only)
    data_path = resolve_preferred_processed_dataset()
    df = pd.read_csv(data_path, low_memory=False).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))
    records_by_row = {int(row["_row_id"]): _smiles_to_graph_record(int(row["_row_id"]), str(row["smiles_canonical"]), float(row["pIC50_median"])) for _, row in df.iterrows()}
    split_frames: dict[str, object] = {
        "random": random_split(df, test_size=0.2, seed=42),
        "scaffold": scaffold_split(df, smiles_col="smiles_canonical", test_size=0.2, seed=42),
    }
    try:
        split_frames["temporal"] = temporal_split(df, year_col="year_max", test_size=0.2)
    except ValueError:
        split_frames["temporal"] = None

    benchmark_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []

    for split_name, split_value in split_frames.items():
        if split_value is None:
            continue
        train_df, test_df = split_value[:2]
        train_idx = train_df["_row_id"].to_numpy(dtype=int)
        test_idx = test_df["_row_id"].to_numpy(dtype=int)
        gnn_result = _fit_gnn_split(records_by_row, train_idx, test_idx, device, args.epochs, args.batch_size, args.hidden_dim, args.learning_rate, args.weight_decay, args.early_patience)
        history_df = gnn_result["history"].copy()
        history_df["split"] = split_name
        history_frames.append(history_df)
        reference_metrics, reference_pred_df = _evaluate_reference_ensemble(df, train_idx, test_idx)
        merged = gnn_result["predictions"].merge(reference_pred_df, on=["row_id", "y_true"], how="inner")
        merged["pred_consensus_blend"] = 0.65 * merged["pred_reference_ensemble"] + 0.35 * merged["pred_gpu_gnn"]
        blend_metrics = _metrics_dict(merged["y_true"].to_numpy(dtype=float), merged["pred_consensus_blend"].to_numpy(dtype=float))
        benchmark_rows.extend(
            [
                {"model": "gpu_graph_regressor", "split": split_name, "device": device_label, **gnn_result["metrics"], "best_epoch": int(gnn_result["best_epoch"]), "train_size": int(len(train_idx)), "test_size": int(len(test_idx))},
                {"model": "multiview_reference", "split": split_name, "device": "cpu_reference", **reference_metrics, "best_epoch": None, "train_size": int(len(train_idx)), "test_size": int(len(test_idx))},
                {"model": "consensus_blend", "split": split_name, "device": device_label, **blend_metrics, "best_epoch": int(gnn_result["best_epoch"]), "train_size": int(len(train_idx)), "test_size": int(len(test_idx))},
            ]
        )
        merged["split"] = split_name
        prediction_frames.append(merged)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df = pd.DataFrame(benchmark_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    benchmark_csv = reports_dir / "gpu_gnn_benchmark.csv"
    predictions_csv = reports_dir / "gpu_gnn_split_predictions.csv"
    history_csv = reports_dir / "gpu_gnn_training_history.csv"
    summary_json = reports_dir / "gpu_gnn_performance_summary.json"
    benchmark_df.to_csv(benchmark_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    history_df.to_csv(history_csv, index=False)
    summary = {
        "dataset_name": dataset_label_from_path(data_path),
        "dataset_path": str(data_path),
        "device": device_label,
        "descriptor_names": DESCRIPTOR_NAMES,
        "splits": benchmark_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved GPU GNN benchmark: {benchmark_csv}")
    print(f"[OK] Saved GPU GNN split predictions: {predictions_csv}")
    print(f"[OK] Saved GPU GNN history: {history_csv}")
    print(f"[OK] Saved GPU GNN summary: {summary_json}")
    if not benchmark_df.empty:
        print(benchmark_df.sort_values(['split', 'rmse']).to_string(index=False))


if __name__ == "__main__":
    main()
