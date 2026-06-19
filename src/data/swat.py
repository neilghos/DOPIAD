from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import HNNCompactEntity


ATTACKS = [
    {'id': 1, 'start_time': '28/12/2015 10:29:14', 'end_time': '28/12/2015 10:44:53'},
    {'id': 2, 'start_time': '28/12/2015 10:51:08', 'end_time': '28/12/2015 10:58:30'},
    {'id': 3, 'start_time': '28/12/2015 11:22:00', 'end_time': '28/12/2015 11:28:22'},
    {'id': 4, 'start_time': '28/12/2015 11:47:39', 'end_time': '28/12/2015 11:54:08'},
    {'id': 6, 'start_time': '28/12/2015 12:00:55', 'end_time': '28/12/2015 12:04:10'},
    {'id': 7, 'start_time': '28/12/2015 12:08:04', 'end_time': '28/12/2015 12:15:33'},
    {'id': 8, 'start_time': '28/12/2015 13:10:10', 'end_time': '28/12/2015 13:26:13'},
    {'id': 10, 'start_time': '28/12/2015 14:16:20', 'end_time': '28/12/2015 14:19:00'},
    {'id': 11, 'start_time': '28/12/2015 14:19:00', 'end_time': '28/12/2015 14:28:20'},
    {'id': 13, 'start_time': '29/12/2015 11:11:25', 'end_time': '29/12/2015 11:15:17'},
    {'id': 14, 'start_time': '29/12/2015 11:35:40', 'end_time': '29/12/2015 11:42:50'},
    {'id': 16, 'start_time': '29/12/2015 11:57:25', 'end_time': '29/12/2015 12:02:00'},
    {'id': 17, 'start_time': '29/12/2015 14:38:12', 'end_time': '29/12/2015 14:50:08'},
    {'id': 19, 'start_time': '29/12/2015 18:10:43', 'end_time': '29/12/2015 18:15:01'},
    {'id': 20, 'start_time': '29/12/2015 18:15:43', 'end_time': '29/12/2015 18:22:17'},
    {'id': 21, 'start_time': '29/12/2015 18:29:58', 'end_time': '29/12/2015 18:42:00'},
    {'id': 22, 'start_time': '29/12/2015 22:55:18', 'end_time': '29/12/2015 23:03:00'},
    {'id': 23, 'start_time': '30/12/2015 01:42:34', 'end_time': '30/12/2015 01:54:10'},
    {'id': 24, 'start_time': '30/12/2015 09:51:08', 'end_time': '30/12/2015 09:56:28'},
    {'id': 25, 'start_time': '30/12/2015 10:01:31', 'end_time': '30/12/2015 10:12:01'},
    {'id': 26, 'start_time': '30/12/2015 17:04:56', 'end_time': '30/12/2015 17:29:00'},
    {'id': 27, 'start_time': '31/12/2015 01:17:08', 'end_time': '31/12/2015 01:45:18'},
    {'id': 28, 'start_time': '31/12/2015 01:45:18', 'end_time': '31/12/2015 11:15:27'},
    {'id': 29, 'start_time': '31/12/2015 15:32:00', 'end_time': '31/12/2015 15:34:00'},
    {'id': 30, 'start_time': '31/12/2015 15:47:02', 'end_time': '31/12/2015 16:07:10'},
    {'id': 31, 'start_time': '31/12/2015 22:05:34', 'end_time': '31/12/2015 22:11:40'},
    {'id': 32, 'start_time': '1/01/2016 10:36:00', 'end_time': '1/01/2016 10:46:36'},
    {'id': 33, 'start_time': '1/01/2016 14:21:12', 'end_time': '1/01/2016 14:28:35'},
    {'id': 34, 'start_time': '1/01/2016 17:12:40', 'end_time': '1/01/2016 17:14:20'},
    {'id': 35, 'start_time': '1/01/2016 17:18:56', 'end_time': '1/01/2016 17:26:56'},
    {'id': 36, 'start_time': '1/01/2016 22:16:01', 'end_time': '1/01/2016 22:25:43'},
    {'id': 37, 'start_time': '2/01/2016 11:17:02', 'end_time': '2/01/2016 11:25:27'},
    {'id': 38, 'start_time': '2/01/2016 11:31:38', 'end_time': '2/01/2016 11:36:18'},
    {'id': 39, 'start_time': '2/01/2016 11:43:48', 'end_time': '2/01/2016 11:50:28'},
    {'id': 40, 'start_time': '2/01/2016 11:51:42', 'end_time': '2/01/2016 11:56:38'},
    {'id': 41, 'start_time': '2/01/2016 13:13:02', 'end_time': '2/01/2016 13:40:56'},
]

for attack in ATTACKS:
    attack['start_time_dt'] = datetime.strptime(attack['start_time'], '%d/%m/%Y %H:%M:%S')
    attack['end_time_dt'] = datetime.strptime(attack['end_time'], '%d/%m/%Y %H:%M:%S')


def _resolve_file(data_root: Path, stem: str) -> Path:
    for ext in ('.csv', '.xlsx', '.xls'):
        path = data_root / f'{stem}{ext}'
        if path.exists():
            return path
    raise FileNotFoundError(f'SWAT file not found for stem {stem!r} in {data_root}')


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, header=None)
    return pd.read_excel(path, header=None)


def _load_matrix(path: Path, input_size: int):
    df = _read_frame(path)
    ts = df.iloc[2:, 0].astype(str).to_numpy()
    data = df.iloc[2:, 1:1 + input_size].to_numpy(dtype=np.float32)
    columns = df.iloc[0, 1:1 + input_size].astype(str).str.strip().tolist()
    return ts, data, columns


def _parse_timestamp(ts: str) -> datetime:
    ts = ts.strip()
    try:
        return datetime.strptime(ts, '%d/%m/%Y %I:%M:%S %p')
    except ValueError:
        return datetime.strptime(ts, '%d/%m/%Y %H:%M:%S')


def _label_from_attacks(ts: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(ts), dtype=np.int64)
    parsed = [_parse_timestamp(str(t)) for t in ts]
    for i, dt in enumerate(parsed):
        for attack in ATTACKS:
            if attack['start_time_dt'] <= dt <= attack['end_time_dt']:
                labels[i] = 1
                break
    return labels


def load_swat_compact(
    data_root: str | Path,
    entity_id: str = 'SWAT_Compact',
    input_size: int = 51,
) -> HNNCompactEntity:
    root = Path(data_root)
    train_path = _resolve_file(root, 'train')
    test_path = _resolve_file(root, 'test')

    _train_ts, train_points, feature_names = _load_matrix(train_path, input_size)
    test_ts, test_points, test_feature_names = _load_matrix(test_path, input_size)
    test_labels = _label_from_attacks(test_ts)

    if feature_names != test_feature_names:
        raise ValueError('SWAT train/test feature name mismatch')
    if test_points.shape[0] != test_labels.shape[0]:
        raise ValueError(
            f'SWAT pooled test shape/label mismatch: test={test_points.shape[0]} labels={test_labels.shape[0]}'
        )

    return HNNCompactEntity(
        entity_id=entity_id,
        feature_names=feature_names,
        train_points=np.asarray(train_points, dtype=np.float32),
        test_points=np.asarray(test_points, dtype=np.float32),
        test_labels=np.asarray(test_labels, dtype=np.int64),
    )
