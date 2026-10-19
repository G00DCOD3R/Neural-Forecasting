"""Shared session loading, trial windows, and forecast metrics.

G-VAR, the graph models, and baselines.py all use these so the splits and scores match.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

GVAR_DIR = Path(__file__).resolve().parent / 'GVAR'
sys.path.insert(0, str(GVAR_DIR))
from data import load_data  # noqa: E402


def load_session(dataset, log_transform, val_frac, test_frac, use_exog):
    """Load a processed session with G-VAR's trial split and activity normalisation."""
    data = load_data(dataset, log_transform, val_frac, test_frac)
    if not use_exog:
        data['U'] = None
        return data
    U = np.load(Path(data['data_dir']) / 'U_exogenous.npy').astype(np.float32)
    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    train = data['split'][data['trial_id']] == 'train'
    U = (U - U[train].mean(0)) / (U[train].std(0) + 1e-6)
    data['U'] = torch.from_numpy(U)
    return data


def split_windows(data, which, history, horizon):
    """Windows that stay inside one trial. x is [W, history + horizon, N]."""
    xs, us = [], []
    length = history + horizon
    for trial in np.flatnonzero(data['split'] == which):
        mask = data['trial_id'] == trial
        x = data['X'][mask]
        if x.shape[0] < length:
            continue
        xs.append(x.unfold(0, length, 1).permute(0, 2, 1))
        if data['U'] is not None:
            us.append(data['U'][mask].unfold(0, length, 1).permute(0, 2, 1))
    if not xs:
        raise RuntimeError(f'No {which} windows of length {length}')
    u = torch.cat(us) if us else None
    return torch.cat(xs), u


class Windows(Dataset):
    def __init__(self, x, u, history):
        self.x, self.u, self.history = x, u, history

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        past, future = self.x[idx, :self.history], self.x[idx, self.history:]
        if self.u is None:
            return past, future
        return past, self.u[idx, :self.history], future


def summarize(pred, target):
    """Per-horizon MSE, RMSE, MAE, mean Pearson and mean R^2. pred/target: [W, H, N]."""
    err = pred - target
    mse = (err ** 2).mean(dim=(0, 2))
    centered_p = pred - pred.mean(0, keepdim=True)
    centered_t = target - target.mean(0, keepdim=True)
    pearson = ((centered_p * centered_t).sum(0) /
               (centered_p.norm(dim=0) * centered_t.norm(dim=0) + 1e-8)).mean(-1)
    r2 = (1 - (err ** 2).sum(0) / ((centered_t ** 2).sum(0) + 1e-8)).mean(-1)
    return {
        'mse': mse.tolist(), 'rmse': mse.sqrt().tolist(), 'mae': err.abs().mean((0, 2)).tolist(),
        'pearson': pearson.tolist(), 'r2': r2.tolist(),
    }


def naive_baselines(windows, history):
    """Copy-last and the training mean, which is zero after normalisation."""
    past, target = windows[:, :history], windows[:, history:]
    return {
        'copy_last': summarize(past[:, -1:].expand_as(target), target),
        'mean': summarize(torch.zeros_like(target), target),
    }
