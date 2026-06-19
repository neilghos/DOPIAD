from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


@dataclass
class HNNCompactEntity:
    entity_id: str
    feature_names: list[str]
    train_points: np.ndarray
    test_points: np.ndarray
    test_labels: np.ndarray


@dataclass
class HNNPreparedData:
    entity_id: str
    feature_names: list[str]
    train_points: np.ndarray
    test_points: np.ndarray
    train_starts: np.ndarray
    val_starts: np.ndarray
    test_starts: np.ndarray
    test_point_labels: np.ndarray
    window_size: int
    stride: int
    test_stride: int
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None


def standardize_train_test(train_points: np.ndarray, test_points: np.ndarray, scaler_mode: str = 'standard'):
    train_points = np.asarray(train_points, dtype=np.float32)
    test_points = np.asarray(test_points, dtype=np.float32)
    if scaler_mode in {'none', 'identity'}:
        return train_points, test_points, None, None
    if scaler_mode != 'standard':
        raise ValueError(f'unsupported scaler_mode: {scaler_mode}')
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_points).astype(np.float32)
    test_scaled = scaler.transform(test_points).astype(np.float32)
    return train_scaled, test_scaled, scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)


def make_window_starts(length: int, window_size: int, stride: int) -> np.ndarray:
    if length < window_size:
        raise ValueError(f'not enough points for one window: len={length} window={window_size}')
    return np.arange(0, length - window_size + 1, stride, dtype=np.int64)


def split_train_val_starts(starts: np.ndarray, val_split: float) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f'val_split must be in [0,1), got {val_split}')
    if val_split == 0.0 or len(starts) < 2:
        return starts, np.zeros((0,), dtype=np.int64)
    cut = int(len(starts) * (1.0 - val_split))
    cut = max(1, min(cut, len(starts) - 1))
    return starts[:cut], starts[cut:]


def truncate_test_labels(test_labels: np.ndarray, num_windows: int, window_size: int, test_stride: int) -> np.ndarray:
    actual_test_len = (num_windows - 1) * test_stride + window_size
    return np.asarray(test_labels[:actual_test_len], dtype=np.int64)


def slice_window(points: np.ndarray, start: int, window_size: int) -> np.ndarray:
    return np.asarray(points[start:start + window_size], dtype=np.float32)


def make_mask_views(anchor: np.ndarray, mask_rates: Sequence[float] = (0.05, 0.15, 0.30, 0.50), seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    views = [anchor.astype(np.float32, copy=True)]
    _w, f = anchor.shape
    for rate in mask_rates:
        mask = rng.random((f,)) < rate
        view = anchor.copy()
        view *= (~mask[None, :])
        views.append(view.astype(np.float32))
    return np.stack(views, axis=0)


def prepare_hnn_compact_data(
    entity: HNNCompactEntity,
    window_size: int,
    stride: int,
    test_stride: int,
    scaler_mode: str = 'standard',
    val_split: float = 0.2,
) -> HNNPreparedData:
    train_scaled, test_scaled, scaler_mean, scaler_scale = standardize_train_test(
        entity.train_points,
        entity.test_points,
        scaler_mode=scaler_mode,
    )
    all_train_starts = make_window_starts(len(train_scaled), window_size, stride)
    train_starts, val_starts = split_train_val_starts(all_train_starts, val_split)
    test_starts = make_window_starts(len(test_scaled), window_size, test_stride)
    test_point_labels = truncate_test_labels(entity.test_labels, len(test_starts), window_size, test_stride)
    return HNNPreparedData(
        entity_id=entity.entity_id,
        feature_names=entity.feature_names,
        train_points=train_scaled,
        test_points=test_scaled,
        train_starts=train_starts,
        val_starts=val_starts,
        test_starts=test_starts,
        test_point_labels=test_point_labels,
        window_size=window_size,
        stride=stride,
        test_stride=test_stride,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )


class HNNTrainDataset(Dataset):
    def __init__(
        self,
        points: np.ndarray,
        starts: np.ndarray,
        window_size: int,
        mask_rates: Sequence[float] = (0.05, 0.15, 0.30, 0.50),
        seed: int | None = None,
    ):
        self.points = points
        self.starts = starts
        self.window_size = window_size
        self.mask_rates = tuple(mask_rates)
        self.seed = seed

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        anchor = slice_window(self.points, start, self.window_size)
        views = make_mask_views(anchor, self.mask_rates, None if self.seed is None else self.seed + idx)
        return {
            'phy_anchor': torch.tensor(anchor, dtype=torch.float32),
            'phy_views': torch.tensor(views, dtype=torch.float32),
            'window_index': start,
            'idx': idx,
        }


class HNNTestDataset(Dataset):
    def __init__(self, points: np.ndarray, starts: np.ndarray, window_size: int, point_labels: np.ndarray):
        self.points = points
        self.starts = starts
        self.window_size = window_size
        self.point_labels = np.asarray(point_labels, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        window = slice_window(self.points, start, self.window_size)
        return {
            'phy': torch.tensor(window, dtype=torch.float32),
            'window_index': start,
            'idx': idx,
            'point_labels': torch.tensor(self.point_labels, dtype=torch.int64),
        }


def build_hnn_datasets(
    entity: HNNCompactEntity,
    window_size: int,
    stride: int,
    test_stride: int,
    scaler_mode: str = 'standard',
    mask_rates: Sequence[float] = (0.05, 0.15, 0.30, 0.50),
    seed: int | None = None,
    val_split: float = 0.2,
):
    prepared = prepare_hnn_compact_data(
        entity=entity,
        window_size=window_size,
        stride=stride,
        test_stride=test_stride,
        scaler_mode=scaler_mode,
        val_split=val_split,
    )
    train_ds = HNNTrainDataset(
        points=prepared.train_points,
        starts=prepared.train_starts,
        window_size=prepared.window_size,
        mask_rates=mask_rates,
        seed=seed,
    )
    val_ds = HNNTrainDataset(
        points=prepared.train_points,
        starts=prepared.val_starts,
        window_size=prepared.window_size,
        mask_rates=mask_rates,
        seed=None if seed is None else seed + 10_000_000,
    )
    test_ds = HNNTestDataset(
        points=prepared.test_points,
        starts=prepared.test_starts,
        window_size=prepared.window_size,
        point_labels=prepared.test_point_labels,
    )
    return train_ds, val_ds, test_ds, prepared
