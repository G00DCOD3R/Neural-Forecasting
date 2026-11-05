# Exogenous inputs for VAR-X / G-VAR-X, row-aligned with X_features.npy, normalised on training rows.
# Video is known ahead (designed stimulus). Behaviour is lagged one row and held during forecasts.
import os

import numpy as np
import torch

BEHAVIOUR_COLUMNS = ('treadmill', 'pupil_0', 'pupil_1', 'pupil_2', 'pupil_3')


# PCA fitted on training frames -> z-scored scores [T, n_pcs], explained variance ratios
def video_pca(frames, train_rows, n_pcs):
    F = torch.from_numpy(frames.reshape(len(frames), -1)).float() / 255
    mean = F[train_rows].mean(0)
    Ft = F[train_rows] - mean
    evals, evecs = torch.linalg.eigh((Ft.T @ Ft).double() / len(Ft))
    evals, evecs = evals.flip(0), evecs.flip(1)  # descending
    scores = (F - mean) @ evecs[:, :n_pcs].float()
    scores = scores / scores[train_rows].std(0)
    return scores.numpy(), (evals[:n_pcs] / evals.sum()).numpy()


# Behaviour columns shifted by one row and z-scored. NaNs raise unless fill_missing (zero-fill + indicator columns).
def behaviour(U, train_rows, columns, fill_missing=False):
    unknown = set(columns) - set(BEHAVIOUR_COLUMNS)
    assert not unknown, f"unknown behaviour columns {unknown}, choose from {BEHAVIOUR_COLUMNS}"
    B = U[:, [BEHAVIOUR_COLUMNS.index(c) for c in columns]].astype(np.float64)
    B = np.concatenate([B[:1], B[:-1]])  # rows are chronological, so this is always the past
    missing = np.isnan(B)
    if missing.any() and not fill_missing:
        bad = [c for c, m in zip(columns, missing.any(0)) if m]
        raise ValueError(f"behaviour columns {bad} contain NaNs, pass fill_missing=True to zero-fill them "
                         f"and add missing-value indicators")
    B = (B - np.nanmean(B[train_rows], 0)) / (np.nanstd(B[train_rows], 0) + 1e-6)
    names = list(columns)
    if missing.any():
        B = np.nan_to_num(B, nan=0.0)
        has = missing.any(0)
        B = np.concatenate([B, missing[:, has].astype(np.float64)], 1)
        names += [f'{c}_missing' for c, h in zip(columns, has) if h]
    return B, names


# -> {'U': [T, E], 'known_ahead': [E], 'names': [E], 'video_explained': [video_pcs]}, E = 0 if nothing requested
def load_exogenous(data_dir, train_rows, video_pcs=0, behaviour_columns=(), fill_missing=False):
    parts, known, names, explained = [], [], [], []
    if video_pcs:
        path = os.path.join(data_dir, 'video_frames.npy')
        assert os.path.exists(path), f"{path} missing, run build_video_features.py first"
        scores, explained = video_pca(np.load(path), train_rows, video_pcs)
        parts.append(scores)
        known += [True] * video_pcs
        names += [f'video_pc{i}' for i in range(video_pcs)]
    if behaviour_columns:
        B, bnames = behaviour(np.load(os.path.join(data_dir, 'U_exogenous.npy')), train_rows,
                              list(behaviour_columns), fill_missing)
        parts.append(B)
        known += [False] * len(bnames)
        names += bnames
    T = len(train_rows)
    U = np.concatenate(parts, 1).astype(np.float32) if parts else np.zeros((T, 0), np.float32)
    return {'U': U, 'known_ahead': np.array(known, bool), 'names': names,
            'video_explained': np.asarray(explained).tolist()}


# Lagged design for targets t0..L-1: [L, E] -> [L - t0, Q*E], row t = [u_t, ..., u_{t-Q+1}]
def lagged(u, Q, t0):
    L = len(u)
    assert t0 >= Q - 1
    blocks = [u[t0 - q:L - q] for q in range(Q)]
    if isinstance(u, torch.Tensor):
        return torch.cat(blocks, 1)
    return np.concatenate(blocks, 1)


# Copy of u [..., W, E] with not-known-ahead columns held at row `origin` (the first target) onwards
def hold_unknown(u, known_ahead, origin):
    u = u.clone() if isinstance(u, torch.Tensor) else u.copy()
    unknown = ~np.asarray(known_ahead, bool)
    if unknown.any():
        cols = np.flatnonzero(unknown)
        u[..., origin + 1:, cols] = u[..., origin:origin + 1, cols]
    return u
