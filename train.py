from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.loader import HNNCompactEntity, build_hnn_datasets, standardize_train_test
from data.smd import load_smd_compact
from data.smap import load_smap_compact
from data.swat import load_swat_compact
from Models.hnn import HNNBranch
from router import route_features
from tsb_metrics import calculate_tsb_metrics
from seedmanager import set_seed


DISCRETE_UNIQUE_THRESHOLD = 25



def load_entity(cfg: DictConfig):
    if cfg.data.name.upper() == 'SMD':
        return load_smd_compact(cfg.data.data_root)
    if cfg.data.name.upper() == 'SMAP':
        return load_smap_compact(cfg.data.data_root)
    if cfg.data.name.upper() == 'SWAT':
        return load_swat_compact(
            cfg.data.data_root,
            input_size=int(cfg.data.get('input_size', 51)),
        )
    raise ValueError(f"Unsupported dataset: {cfg.data.name}")




def select_entity_by_sensor_mode(cfg: DictConfig, entity: HNNCompactEntity) -> HNNCompactEntity:
    sensor_mode = str(cfg.experiment.sensor_mode if 'experiment' in cfg else 'all')
    if sensor_mode == 'all':
        return entity
    if sensor_mode not in {'correlated', 'correlated_minus_piecewise'}:
        raise NotImplementedError(f"sensor_mode {sensor_mode!r} is not wired yet")

    train_scaled, test_scaled, _, _ = standardize_train_test(
        entity.train_points,
        entity.test_points,
        scaler_mode=str(cfg.data.scaler_mode),
    )
    (_, _, _, _), topo, _ = route_features(train_scaled, test_scaled)
    idx = np.asarray(topo.idx_phy, dtype=np.int64)
    if idx.size == 0:
        raise ValueError('correlated routing produced zero sensors')

    if sensor_mode == 'correlated_minus_piecewise':
        idx_discrete = np.array(
            [i for i in idx if np.unique(entity.train_points[:, i]).size < DISCRETE_UNIQUE_THRESHOLD],
            dtype=np.int64,
        )
        idx = np.setdiff1d(idx, idx_discrete)
        print(
            f'piecewise_filter: removed={len(idx_discrete)} kept={len(idx)} '
            f'threshold={DISCRETE_UNIQUE_THRESHOLD}'
        )
        print(f'piecewise_indices: {idx_discrete.tolist()}')
        if idx.size == 0:
            raise ValueError('piecewise filter removed all correlated sensors')

    feature_names = [entity.feature_names[int(i)] for i in idx]
    return HNNCompactEntity(
        entity_id=f"{entity.entity_id}_{sensor_mode}",
        feature_names=feature_names,
        train_points=np.asarray(entity.train_points[:, idx], dtype=np.float32),
        test_points=np.asarray(entity.test_points[:, idx], dtype=np.float32),
        test_labels=entity.test_labels,
    )

def build_model(cfg: DictConfig, input_dim: int) -> HNNBranch:
    return HNNBranch(
        input_dim=input_dim,
        window_size=cfg.data.window_size,
        latent_dim=cfg.model.latent_dim,
        steps=cfg.model.steps,
        dt=cfg.model.dt,
        hamiltonian_hidden_dim=cfg.model.hamiltonian_hidden_dim,
        decoder_hidden_dim=cfg.model.decoder_hidden_dim,
    )


def reconstruction_loss(anchor: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return torch.mean((anchor - recon) ** 2)


def run_train_epoch(model: HNNBranch, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch_idx: int, total_epochs: int) -> float:
    model.train()
    losses = []
    bar = tqdm(loader, desc=f'Train {epoch_idx + 1}/{total_epochs}', unit='batch')
    for batch in bar:
        phy_anchor = batch['phy_anchor'].to(device)
        optimizer.zero_grad(set_to_none=True)
        _, recon = model(phy_anchor)
        loss = reconstruction_loss(phy_anchor, recon)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)
        bar.set_postfix({'recon': f'{loss_val:.6f}'})
    return float(np.mean(losses)) if losses else float('inf')


def run_val_epoch(model: HNNBranch, loader: DataLoader, device: torch.device, epoch_idx: int, total_epochs: int) -> float:
    model.eval()
    losses = []
    bar = tqdm(loader, desc=f'Val   {epoch_idx + 1}/{total_epochs}', unit='batch')
    for batch in bar:
        phy_anchor = batch['phy_anchor'].to(device)
        _, recon = model(phy_anchor)
        loss = reconstruction_loss(phy_anchor, recon)
        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)
        bar.set_postfix({'mse': f'{loss_val:.6f}'})
    return float(np.mean(losses)) if losses else float('inf')


def tail_stitch_window_scores(window_scores: np.ndarray, starts: np.ndarray, total_len: int, window_size: int, stride: int) -> np.ndarray:
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


def reconstruct_test_windows(model: HNNBranch, test_loader: DataLoader, device: torch.device):
    model.eval()
    window_errs = []
    starts = []
    bar = tqdm(test_loader, desc='Test reconstruct', unit='batch')
    for batch in bar:
        phy = batch['phy'].to(device)
        _, recon = model(phy)
        err = torch.mean((phy - recon) ** 2, dim=-1)
        window_errs.append(err.detach().cpu().numpy().astype(np.float32))
        starts.append(batch['window_index'].cpu().numpy().astype(np.int64))
    return np.concatenate(window_errs, axis=0), np.concatenate(starts, axis=0)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    print(OmegaConf.to_yaml(cfg))

    entity = load_entity(cfg)
    entity = select_entity_by_sensor_mode(cfg, entity)
    train_ds, val_ds, test_ds, prepared = build_hnn_datasets(
        entity=entity,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
        test_stride=int(cfg.data.test_stride),
        scaler_mode=str(cfg.data.scaler_mode),
        mask_rates=tuple(cfg.data.mask_rates),
        seed=int(cfg.seed),
        val_split=float(cfg.data.val_split),
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg.data.train_batch_size), shuffle=True, num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.data.eval_batch_size), shuffle=False, num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory), drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.data.eval_batch_size), shuffle=False, num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory), drop_last=False)

    input_dim = prepared.train_points.shape[1]
    requested_device = str(cfg.device)
    device = torch.device(requested_device if (requested_device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    model = build_model(cfg, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.model.lr))

    print(f'entity_id: {prepared.entity_id}')
    print(f'sensor_mode: {cfg.experiment.sensor_mode if "experiment" in cfg else "all"}')
    print(f'input_dim: {input_dim}')
    print(f'train_windows: {len(train_ds)}')
    print(f'val_windows: {len(val_ds)}')
    print(f'test_windows: {len(test_ds)}')
    print(f'test_label_len: {len(prepared.test_point_labels)}')

    best_val = float('inf')
    best_epoch = -1
    wait = 0
    ckpt_path = Path('best_model.pt')

    for epoch in range(int(cfg.model.epochs)):
        train_loss = run_train_epoch(model, train_loader, optimizer, device, epoch, int(cfg.model.epochs))
        val_loss = run_val_epoch(model, val_loader, device, epoch, int(cfg.model.epochs))
        print(f'Epoch {epoch + 1}: train_recon={train_loss:.6f} val_mse={val_loss:.6f} best_val={best_val:.6f}')
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            wait = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': best_epoch, 'best_val_mse': best_val, 'cfg': OmegaConf.to_container(cfg, resolve=True)}, ckpt_path)
            print(f'Saved best model to {ckpt_path} at epoch {best_epoch}')
        else:
            wait += 1
            if wait >= int(cfg.model.patience):
                print(f'Early stopping at epoch {epoch + 1} (patience={int(cfg.model.patience)})')
                break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f'Loaded best model from epoch {state.get("epoch", -1)} with val_mse={state.get("best_val_mse", float("nan")):.6f}')

    window_scores, starts = reconstruct_test_windows(model, test_loader, device)
    point_scores = tail_stitch_window_scores(
        window_scores=window_scores,
        starts=starts,
        total_len=len(prepared.test_point_labels),
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.test_stride),
    )
    metrics = calculate_tsb_metrics(point_scores, prepared.test_point_labels)

    print(f'best_epoch: {best_epoch}')
    print(f'window_scores: {tuple(window_scores.shape)}')
    print(f'point_scores: {tuple(point_scores.shape)}')
    print('metrics:')
    for key in ['auc', 'prauc', 'p_best', 'r_best', 'f1_best', 'vusaucc', 'vuspr', 'aff_p', 'aff_r', 'aff1']:
        print(f'  {key}: {metrics[key]:.6f}')


if __name__ == '__main__':
    main()
