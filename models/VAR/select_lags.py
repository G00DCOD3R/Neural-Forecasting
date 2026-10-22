"""Select VAR activity lag order on validation only and save the selected fit."""
import argparse
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
import scipy

try:
    from .data import load_session
    from .lag_selection import select_lags
    from .train import sha256
except ImportError:
    from data import load_session
    from lag_selection import select_lags
    from train import sha256

RUNS_DIR = Path(__file__).resolve().parent/'runs'


def run(dataset='cave_proofread', lags=(1, 2, 4), lambdas=(0.01, 0.1, 1),
        context=15, batch_size=256, output_dir=None, verbose=True):
    output = Path(output_dir or RUNS_DIR/f'lag_selection_{uuid4().hex[:8]}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty output directory')
    lags, lambdas = list(lags), list(lambdas)
    started = time.perf_counter()
    data = load_session(dataset)
    if verbose:
        print(f'Loaded {tuple(data.X.shape)} activity; {data.summary()["trials_per_split"]}', flush=True)
        print(f'VALIDATION SELECTION ONLY: common train/validation context={context}, horizon=1', flush=True)
        print('Original VAR1 used earlier training targets too; this is a new controlled comparison.', flush=True)
    model, grid = select_lags(data.trials('train'), data.trials('val'), lags, lambdas,
                              context, batch_size, verbose)
    results = {'phase': 'validation_selection_only',
               'config': {'dataset': str(dataset), 'lags': lags, 'lambdas': lambdas,
                          'context': context, 'training_start': context, 'horizon': 1,
                          'batch_size': batch_size},
               'selected': {'P': model.P, 'lambda': model.lam,
                            'parameter_count': model.parameter_count,
                            'fitted_training_pairs': model.n_train_pairs},
               'grid': grid, 'data': {**data.summary(),
                   'input_sha256': {name: sha256(data.directory/name)
                                    for name in ('X_features.npy', 'conditions.csv')},
                   'fit_trial_ids': np.flatnonzero(data.split=='train').tolist(),
                   'validation_trial_ids': np.flatnonzero(data.split=='val').tolist()},
               'metric_space': 'training-standardized log1p(clipped activity)',
               'r2_convention': 'mean per-neuron 1-SSE/(target_SS+1e-8), including constants',
               'source_sha256': {name: sha256(Path(__file__).parent/name) for name in
                                 ('var_p.py', 'lag_selection.py', 'select_lags.py', 'data.py', 'evaluation.py')},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'elapsed_seconds': time.perf_counter()-started}
    serialized = json.dumps(results, indent=2, allow_nan=False)
    lines = ['VALIDATION SELECTION ONLY - no test forecasts computed',
             f'Common training and evaluation context: {context}; one-step targets',
             '', ' P     lambda     parameters      train MSE        val MSE     val Pearson']
    lines += [f'{r["P"]:2d} {r["lambda"]:10g} {r["parameter_count"]:14,d} '
              f'{r["train"]["mse"][0]:14.9f} {r["validation"]["mse"][0]:14.9f} '
              f'{r["validation"]["pearson"][0]:15.6f}' for r in grid]
    lines += ['', f'Selected P={model.P}, lambda={model.lam:g} by lowest one-step validation MSE.',
              'The selected checkpoint is ready for later evaluation; test data was not scored.']
    summary = '\n'.join(lines)+'\n'
    output.mkdir(parents=True, exist_ok=True)
    (output/'selection.json').write_text(serialized+'\n')
    (output/'selection_summary.txt').write_text(summary)
    np.savez_compressed(output/'model.npz', P=model.P, lam=model.lam,
                        coefficients=model.coefficients, intercept=model.intercept,
                        n_train_pairs=model.n_train_pairs, training_start=model.training_start,
                        mu=data.mu, sd=data.sd, log_transform=data.log_transform)
    if verbose:
        print('\n'+summary, end='')
        print('Saved to', output, flush=True)
    return model, results, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.01, 0.1, 1])
    parser.add_argument('--context', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=Path)
    args = vars(parser.parse_args())
    try:
        run(**args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f'Lag selection: {exc}\n')


if __name__ == '__main__':
    main()
