"""Verify the processed-file interface using small synthetic recordings."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

try:
    from .data import load_session
    from .synthetic_example import generate_trials
except ImportError:
    from data import load_session
    from synthetic_example import generate_trials


def write_synthetic_session(directory, seed=42, count=20, length=500):
    """Write tiny synthetic processed files for examples/tests, never real data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trials, _, _ = generate_trials(seed=seed, count=count, length=length)
    X = np.vstack(trials).astype(np.float32)
    rows = []
    for trial_id, trial in enumerate(trials):
        for frame in range(len(trial)):
            rows.append({'trial_id': trial_id, 'trial_name': f'trial_{trial_id}',
                         'condition_hash': 'example_A' if trial_id % 2 == 0 else 'example_B',
                         'time': (trial_id * (len(trial)+2) + frame) / 7.5})
    np.save(directory/'X_features.npy', X)
    pd.DataFrame(rows).to_csv(directory/'conditions.csv', index=False)


def main():
    with tempfile.TemporaryDirectory(prefix='var_data_example_') as temporary:
        directory = Path(temporary)
        write_synthetic_session(directory)
        data = load_session(directory)
        print('SYNTHETIC DATA INTERFACE ONLY: temporary files, not MICrONS')
        summary = data.summary()
        for key in ('shape', 'trials_per_split', 'rows_per_split', 'log_transform', 'negative_values'):
            print(f'{key}: {summary[key]}')
        train = np.vstack(data.trials('train'))
        print('Normalized training mean per neuron:', np.round(train.mean(0), 5))
        print('Normalized training std per neuron: ', np.round(train.std(0), 5))
        print('Distinct stimulus IDs: 2; distinct trial IDs: 20')
        print('Validation and test use the same normalization parameters fitted on training only.')
        print('Temporary synthetic files are removed when this example finishes.')


if __name__ == '__main__':
    main()
