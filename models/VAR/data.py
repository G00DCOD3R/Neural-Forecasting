"""Load chronological processed recordings and normalize using training only."""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from exogenous import load_exogenous  # noqa: E402
DATASETS = {
    'cave_all': REPO_ROOT / 'processed' / 'session_4_7_cave_all',
    'cave_proofread': REPO_ROOT / 'processed' / 'session_4_7_cave_proofread',
}


@dataclass
class SessionData:
    X: np.ndarray
    trial_id: np.ndarray
    time: np.ndarray
    trial_start: np.ndarray
    trial_bounds: np.ndarray
    split: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    log_transform: bool
    negative_values: int
    directory: Path
    # exogenous inputs [time, E] (see exogenous.py), E = 0 by default
    U: np.ndarray = None
    known_ahead: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    exo_names: list = field(default_factory=list)

    def trials(self, which):
        """Return array views of individual trials in one chronological split."""
        if which not in ('train', 'val', 'test'):
            raise ValueError('which must be train, val or test')
        return [self.X[start:stop] for (start, stop), label in zip(self.trial_bounds, self.split)
                if label == which]

    # [time, E] views of the exogenous inputs, aligned with trials(which)
    def exo_trials(self, which):
        if which not in ('train', 'val', 'test'):
            raise ValueError('which must be train, val or test')
        return [self.U[start:stop] for (start, stop), label in zip(self.trial_bounds, self.split)
                if label == which]

    def summary(self):
        return {'shape': list(self.X.shape), 'directory': str(self.directory),
                'log_transform': self.log_transform, 'negative_values': self.negative_values,
                'trials_per_split': {s: int((self.split == s).sum()) for s in ('train', 'val', 'test')},
                'rows_per_split': {s: sum(len(x) for x in self.trials(s)) for s in ('train', 'val', 'test')}}


# video_pcs, behaviour, fill_missing select exogenous inputs (see exogenous.py)
def load_session(dataset='cave_all', log_transform=True, val_frac=0.15, test_frac=0.15, video_pcs=0,
                 behaviour=(), fill_missing=False):
    """Read X_features.npy and timestamp-aware conditions.csv.

    Split thresholds, log1p/clipping, float32 statistics and std+1e-6 match
    Andrei's GVAR data loader. No graph files, CAVE token or raw HDF5 is needed.
    """
    directory = Path(DATASETS.get(str(dataset), dataset)).resolve()
    x_path, metadata_path = directory / 'X_features.npy', directory / 'conditions.csv'
    if not x_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f'Missing X_features.npy or conditions.csv in {directory}. '
            'Obtain the processed session or build it with the timestamp-corrected builder.'
        )
    if (not np.isfinite(val_frac) or not np.isfinite(test_frac) or val_frac <= 0
            or test_frac <= 0 or val_frac + test_frac >= 1):
        raise ValueError('Validation/test fractions must be positive and sum to less than one')
    raw = np.load(x_path, mmap_mode='r', allow_pickle=False)
    if (raw.ndim != 2 or not all(raw.shape) or not np.issubdtype(raw.dtype, np.number)
            or np.iscomplexobj(raw)):
        raise ValueError('X_features.npy must be a nonempty real [time, neurons] array')
    if not np.isfinite(raw).all():
        raise ValueError('Activity contains NaN/Inf; agree on a common cleaning policy before comparison')
    rows = pd.read_csv(metadata_path)
    if {'trial_id', 'time'} - set(rows):
        raise ValueError('conditions.csv needs trial_id and time; rebuild using the timestamp-corrected builder')
    if len(rows) != len(raw):
        raise ValueError('There must be one metadata row for every activity timestep')
    ids = pd.to_numeric(rows['trial_id'], errors='raise').to_numpy(dtype=np.float64)
    times = pd.to_numeric(rows['time'], errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(ids).all() or not np.array_equal(ids, ids.astype(np.int64)):
        raise ValueError('trial_id must contain finite integers')
    ids = ids.astype(np.int64)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('Recording timestamps must be finite and chronologically ordered')
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    stops = np.r_[starts[1:], len(ids)]
    if len(starts) < 3 or not np.array_equal(ids[starts], np.arange(len(starts))):
        raise ValueError('Trials must be contiguous, numbered 0..n-1, with at least three trials')
    if np.any((ids[1:] == ids[:-1]) & (np.diff(times) <= 0)):
        raise ValueError('Time must strictly increase within each trial')
    trial_start = times[starts]
    if trial_start[-1] <= trial_start[0]:
        raise ValueError('Trial start times must span a positive duration')

    # Keep whole trials together. This is GVAR's trial-start-time convention,
    # so actual row proportions need not be exactly 70/15/15.
    relative_time = (trial_start - trial_start[0]) / (trial_start[-1] - trial_start[0])
    split = np.where(relative_time >= 1-test_frac, 'test',
                     np.where(relative_time >= 1-test_frac-val_frac, 'val', 'train'))
    if any(not np.any(split == s) for s in ('train', 'val', 'test')):
        raise ValueError('At least one trial is needed in each split; inspect session duration and fractions')

    negative_values = int(np.count_nonzero(raw < 0))
    X = raw.astype(np.float32)
    if log_transform:
        X = np.log1p(np.clip(X, 0, None))
    train_rows = split[ids] == 'train'
    mu = X[train_rows].mean(0)
    sd = X[train_rows].std(0) + 1e-6
    X = (X - mu) / sd
    if not np.isfinite(X).all():
        raise ValueError('Activity became nonfinite during float32 preprocessing')
    exo = load_exogenous(directory, train_rows, video_pcs, tuple(behaviour), fill_missing)
    return SessionData(X, ids, times, trial_start, np.column_stack([starts, stops]),
                       split, mu, sd, bool(log_transform), negative_values, directory,
                       exo['U'].astype(np.float64), exo['known_ahead'], exo['names'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='cave_all', help='cave_all, cave_proofread or a processed folder')
    parser.add_argument('--no-log-transform', action='store_true')
    args = parser.parse_args()
    try:
        data = load_session(args.dataset, log_transform=not args.no_log_transform)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Data: {exc}\n')
    print('Processed session loaded successfully:')
    for name, value in data.summary().items():
        print(f'{name}: {value}')


if __name__ == '__main__':
    main()
