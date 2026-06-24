from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class ScoredSeries:
    point_scores: np.ndarray
    aligned_labels: np.ndarray
    window_scores: np.ndarray
    starts: np.ndarray


SCORER_MODE_ALIASES = {
    "endpoint_stich": "endpoint_stitch",
}


def normalize_scorer_mode(mode: str) -> str:
    normalized = str(mode).strip()
    return SCORER_MODE_ALIASES.get(normalized, normalized)


def _validate_common(
    window_size: int,
    test_stride: int,
    labels: np.ndarray,
    starts: np.ndarray,
) -> None:
    if window_size <= 1:
        raise ValueError("window_size must be > 1 for causal scorer alignment")
    if test_stride != 1:
        raise ValueError("test_stride must be 1 for strict causal scorer alignment")
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("starts must be a non-empty 1D array")
    expected = np.arange(starts[0], starts[0] + len(starts), dtype=np.int64)
    if not np.array_equal(starts, expected):
        raise ValueError("window starts must be contiguous with stride 1 for strict alignment")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")


def _tail_stitch_window_scores(
    window_scores: np.ndarray,
    starts: np.ndarray,
    total_len: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    scores = np.zeros((total_len,), dtype=np.float32)
    counts = np.zeros((total_len,), dtype=np.float32)
    prev_start = None
    for i, start in enumerate(starts):
        start = int(start)
        end = min(start + window_size, total_len)
        if end <= start:
            prev_start = start
            continue
        full_window = prev_start is None or (start - prev_start > stride) or (stride >= window_size)
        slice_start = start if full_window else max(start, end - stride)
        if slice_start >= end:
            prev_start = start
            continue
        offset = slice_start - start
        slice_len = end - slice_start
        scores[slice_start:end] += window_scores[i, offset:offset + slice_len]
        counts[slice_start:end] += 1.0
        prev_start = start
    return scores / np.maximum(counts, 1.0)


def _endpoint_stitch(
    window_scores: np.ndarray,
    labels: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    test_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_common(window_size, test_stride, labels, starts)
    total_len = starts[-1] + window_size
    if total_len > len(labels):
        raise ValueError(
            f"labels shorter than required by stitched windows: labels={len(labels)} required={total_len}"
        )
    point_scores = _tail_stitch_window_scores(
        window_scores=window_scores,
        starts=starts,
        total_len=total_len,
        window_size=window_size,
        stride=test_stride,
    )
    aligned_labels = np.asarray(labels[:total_len], dtype=np.int64)
    if len(point_scores) != len(aligned_labels):
        raise ValueError("endpoint_stitch produced score/label length mismatch")
    return point_scores, aligned_labels


def _endpoint_mean(
    window_scores: np.ndarray,
    labels: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    test_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_common(window_size, test_stride, labels, starts)
    scalar_scores = np.asarray(window_scores.mean(axis=1), dtype=np.float32)
    endpoint_indices = starts + (window_size - 1)
    if endpoint_indices[-1] >= len(labels):
        raise ValueError(
            f"labels shorter than required by endpoint indices: labels={len(labels)} "
            f"last_endpoint={int(endpoint_indices[-1])}"
        )
    aligned_labels = np.asarray(labels[endpoint_indices], dtype=np.int64)
    if len(scalar_scores) != len(aligned_labels):
        raise ValueError("endpoint_mean produced score/label length mismatch")
    return scalar_scores, aligned_labels


def collect_window_scores(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    window_errs = []
    starts = []
    bar = tqdm(test_loader, desc="Test reconstruct", unit="batch")
    with torch.enable_grad():
        for batch in bar:
            phy = batch["phy"].to(device)
            _, recon = model(phy)
            err = torch.mean((phy - recon) ** 2, dim=-1)
            window_errs.append(err.detach().cpu().numpy().astype(np.float32))
            starts.append(batch["window_index"].cpu().numpy().astype(np.int64))
    return np.concatenate(window_errs, axis=0), np.concatenate(starts, axis=0)


def score_reconstruction_windows(
    window_scores: np.ndarray,
    labels: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    test_stride: int,
    mode: str,
) -> ScoredSeries:
    if window_scores.ndim != 2:
        raise ValueError(f"window_scores must be [N, W], got shape={tuple(window_scores.shape)}")
    if window_scores.shape[1] != window_size:
        raise ValueError(
            f"window score width must equal window_size: width={window_scores.shape[1]} window_size={window_size}"
        )

    mode = normalize_scorer_mode(mode)

    if mode == "endpoint_stitch":
        point_scores, aligned_labels = _endpoint_stitch(
            window_scores=window_scores,
            labels=labels,
            starts=starts,
            window_size=window_size,
            test_stride=test_stride,
        )
    elif mode == "endpoint_mean":
        point_scores, aligned_labels = _endpoint_mean(
            window_scores=window_scores,
            labels=labels,
            starts=starts,
            window_size=window_size,
            test_stride=test_stride,
        )
    else:
        raise ValueError(f"unknown scorer mode: {mode}")

    return ScoredSeries(
        point_scores=point_scores,
        aligned_labels=aligned_labels,
        window_scores=window_scores,
        starts=starts,
    )


def score_model_reconstructions(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    labels: np.ndarray,
    window_size: int,
    test_stride: int,
    mode: str,
) -> ScoredSeries:
    window_scores, starts = collect_window_scores(
        model=model,
        test_loader=test_loader,
        device=device,
    )
    return score_reconstruction_windows(
        window_scores=window_scores,
        labels=np.asarray(labels, dtype=np.int64),
        starts=starts,
        window_size=window_size,
        test_stride=test_stride,
        mode=normalize_scorer_mode(mode),
    )


def save_score_cache(
    path: str | Path,
    *,
    window_scores: np.ndarray,
    starts: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    test_stride: int,
    run_name: str,
    dataset: str,
    entity_id: str,
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_scores=np.asarray(window_scores, dtype=np.float32),
        starts=np.asarray(starts, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        window_size=np.asarray([int(window_size)], dtype=np.int64),
        test_stride=np.asarray([int(test_stride)], dtype=np.int64),
        run_name=np.asarray([str(run_name)]),
        dataset=np.asarray([str(dataset)]),
        entity_id=np.asarray([str(entity_id)]),
        metadata_json=np.asarray([json.dumps(metadata or {}, sort_keys=True)]),
    )


def load_score_cache(path: str | Path) -> dict[str, np.ndarray | str | int]:
    cache = np.load(Path(path), allow_pickle=False)
    return {
        "window_scores": np.asarray(cache["window_scores"], dtype=np.float32),
        "starts": np.asarray(cache["starts"], dtype=np.int64),
        "labels": np.asarray(cache["labels"], dtype=np.int64),
        "window_size": int(np.asarray(cache["window_size"]).item()),
        "test_stride": int(np.asarray(cache["test_stride"]).item()),
        "run_name": str(np.asarray(cache["run_name"]).item()),
        "dataset": str(np.asarray(cache["dataset"]).item()),
        "entity_id": str(np.asarray(cache["entity_id"]).item()),
        "metadata": json.loads(str(np.asarray(cache["metadata_json"]).item())),
    }
