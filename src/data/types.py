 from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CompactEntityData:
    entity_id: str
    feature_names: list[str]
    train_points: np.ndarray
    test_points: np.ndarray
    test_labels: np.ndarray
    val_points: Optional[np.ndarray] = None


@dataclass
class CompactPreparedData:
    entity_id: str
    feature_names: list[str]
    train_windows: np.ndarray
    val_windows: np.ndarray
    test_windows: np.ndarray
    test_point_labels: np.ndarray
    train_window_indices: np.ndarray
    val_window_indices: np.ndarray
    test_window_indices: np.ndarray
    train_points: np.ndarray
    val_points: np.ndarray | None
    test_points: np.ndarray
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
