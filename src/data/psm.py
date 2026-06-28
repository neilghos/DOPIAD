from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import HNNCompactEntity


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PSM file not found: {path}")
    return pd.read_csv(path).ffill().bfill()


def load_psm_compact(data_root: str | Path, entity_id: str = "PSM_Compact") -> HNNCompactEntity:
    root = Path(data_root)
    train_df = _load_frame(root / "train.csv")
    test_df = _load_frame(root / "test.csv")
    label_df = _load_frame(root / "test_label.csv")

    train_points = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    test_points = test_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    test_labels = label_df.iloc[:, 1:].to_numpy(dtype=np.int64).reshape(-1)

    if train_points.shape[1] != test_points.shape[1]:
        raise ValueError(
            f"PSM train/test feature mismatch: train={train_points.shape[1]} test={test_points.shape[1]}"
        )
    if test_points.shape[0] != test_labels.shape[0]:
        raise ValueError(
            f"PSM pooled test shape/label mismatch: test={test_points.shape[0]} labels={test_labels.shape[0]}"
        )

    feature_names = [str(x).strip() for x in train_df.columns[1:]]
    return HNNCompactEntity(
        entity_id=entity_id,
        feature_names=feature_names,
        train_points=np.asarray(train_points, dtype=np.float32),
        test_points=np.asarray(test_points, dtype=np.float32),
        test_labels=np.asarray(test_labels, dtype=np.int64),
    )
