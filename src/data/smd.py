from __future__ import annotations

from pathlib import Path

import numpy as np

from data.loader import HNNCompactEntity


def _list_smd_ids(data_root: Path) -> list[str]:
    train_dir = data_root / 'train'
    if not train_dir.is_dir():
        raise FileNotFoundError(f'SMD train directory not found: {train_dir}')
    return sorted(p.stem for p in train_dir.glob('*.txt'))


def _load_txt_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=',', dtype=np.float32)


def _stack_smd_split(data_root: Path, subset: str, machine_ids: list[str]) -> np.ndarray:
    arrays = [_load_txt_matrix(data_root / subset / f'{machine_id}.txt') for machine_id in machine_ids]
    if not arrays:
        raise ValueError(f'No SMD files found for split {subset!r} in {data_root}')
    return np.concatenate(arrays, axis=0).astype(np.float32)


def _stack_smd_labels(data_root: Path, machine_ids: list[str]) -> np.ndarray:
    arrays = [
        _load_txt_matrix(data_root / 'test_label' / f'{machine_id}.txt').reshape(-1)
        for machine_id in machine_ids
    ]
    if not arrays:
        raise ValueError(f'No SMD label files found in {data_root}')
    return np.concatenate(arrays, axis=0).astype(np.int64)


def load_smd_compact(data_root: str | Path, entity_id: str = 'SMD_Compact') -> HNNCompactEntity:
    root = Path(data_root)
    machine_ids = _list_smd_ids(root)
    train_points = _stack_smd_split(root, 'train', machine_ids)
    test_points = _stack_smd_split(root, 'test', machine_ids)
    test_labels = _stack_smd_labels(root, machine_ids)
    if test_points.shape[0] != test_labels.shape[0]:
        raise ValueError(f'SMD pooled test shape/label mismatch: test={test_points.shape[0]} labels={test_labels.shape[0]}')
    feature_names = [f'sensor_{i:02d}' for i in range(train_points.shape[1])]
    return HNNCompactEntity(
        entity_id=entity_id,
        feature_names=feature_names,
        train_points=train_points,
        test_points=test_points,
        test_labels=test_labels,
    )
