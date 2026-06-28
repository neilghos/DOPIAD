from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import HNNCompactEntity


def _label_csv_path(data_root: Path) -> Path:
    for name in ("labeled_anomalies.csv", "labelled_anomalies.csv"):
        path = data_root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Telemetry label CSV not found in {data_root}")


def _list_channel_ids(data_root: Path) -> list[str]:
    train_dir = data_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Telemetry train directory not found: {train_dir}")
    return sorted(p.stem for p in train_dir.glob("*.npy"))


def _load_channel_arrays(data_root: Path, channel_ids: list[str]):
    train_arrays = []
    test_arrays = []
    test_lengths = []
    for channel_id in channel_ids:
        train_raw = np.load(data_root / "train" / f"{channel_id}.npy").astype(np.float32)
        test_raw = np.load(data_root / "test" / f"{channel_id}.npy").astype(np.float32)
        train_arrays.append(train_raw)
        test_arrays.append(test_raw)
        test_lengths.append(test_raw.shape[0])
    if not train_arrays:
        raise ValueError(f"No telemetry channels found in {data_root}")
    return train_arrays, test_arrays, test_lengths


def _stack_labels(data_root: Path, channel_ids: list[str], test_lengths: list[int]) -> np.ndarray:
    df = pd.read_csv(_label_csv_path(data_root))
    labels = []
    for channel_id, test_len in zip(channel_ids, test_lengths):
        query = df[df["chan_id"] == channel_id]
        if query.empty:
            raise ValueError(f"Telemetry channel {channel_id} not found in labels CSV")
        channel_labels = np.zeros(test_len, dtype=np.int64)
        for _, row in query.iterrows():
            anomaly_spans = ast.literal_eval(row["anomaly_sequences"])
            for start, end in anomaly_spans:
                channel_labels[start:end + 1] = 1
        labels.append(channel_labels)
    return np.concatenate(labels, axis=0).astype(np.int64)


def load_msl_compact(data_root: str | Path, entity_id: str = "MSL_Compact") -> HNNCompactEntity:
    root = Path(data_root)
    channel_ids = _list_channel_ids(root)
    train_arrays, test_arrays, test_lengths = _load_channel_arrays(root, channel_ids)
    train_points = np.concatenate(train_arrays, axis=0).astype(np.float32)
    test_points = np.concatenate(test_arrays, axis=0).astype(np.float32)
    test_labels = _stack_labels(root, channel_ids, test_lengths)
    if test_points.shape[0] != test_labels.shape[0]:
        raise ValueError(
            f"MSL pooled test shape/label mismatch: test={test_points.shape[0]} labels={test_labels.shape[0]}"
        )
    feature_names = [f"sensor_{i:02d}" for i in range(train_points.shape[1])]
    return HNNCompactEntity(
        entity_id=entity_id,
        feature_names=feature_names,
        train_points=train_points,
        test_points=test_points,
        test_labels=test_labels,
    )
