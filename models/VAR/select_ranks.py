"""Compare reduced-rank and unrestricted ridge VAR using validation only."""
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
    from .lag_selection import evaluate
    from .reduced_rank import RankProjection, ReducedRankVAR
    from .train import sha256
    from .var_p import VAR
except ImportError:
    from data import load_session
    from lag_selection import evaluate
    from reduced_rank import RankProjection, ReducedRankVAR
    from train import sha256
    from var_p import VAR

RUNS_DIR = Path(__file__).resolve().parent/'runs'


def select_ranks(train_trials, validation_trials, P=2, ranks=(16, 32, 64, 128, 256, 512),
                 lambdas=(0.01, 0.1, 1), context=15, batch_size=256, verbose=True):
    train_trials, validation_trials = list(train_trials), list(validation_trials)
    ranks, lambdas = list(ranks), list(lambdas)
    base = VAR(P)
    if not isinstance(context, (int, np.integer)) or context < P:
        raise ValueError('Context must be an integer >= P')
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('Batch size must be a positive integer')
    if not ranks or any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) or r < 1 for r in ranks):
        raise ValueError('Provide positive integer ranks')
    if not lambdas or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError('Provide finite positive ridge penalties')
    stats = base.statistics(train_trials, start=context)
    if max(ranks) > stats.num_neurons:
        raise ValueError('A requested rank exceeds the number of output neurons')
    if verbose:
        print(f'Fitting targets per candidate: {stats.count:,}; dense reference included for each lambda', flush=True)
    best_model, best_score, grid = None, float('inf'), []
    for lam in lambdas:
        if verbose:
            print(f'Ridge fit and training-only rank basis: lambda={lam:g}...', flush=True)
        projection = RankProjection.from_statistics(stats, lam)
        for rank in [None, *ranks]:
            model = projection.full_model if rank is None else projection.model(rank)
            training = evaluate(model, train_trials, context, batch_size=batch_size)
            validation = evaluate(model, validation_trials, context, batch_size=batch_size)
            dimension = model.parameter_count if rank is None else model.coefficient_dimension
            row = {'P': int(P), 'rank': None if rank is None else int(rank), 'lambda': float(lam),
                   'family': 'unrestricted_ridge' if rank is None else 'reduced_rank_ridge',
                   'stored_parameter_count': model.parameter_count, 'coefficient_dimension': dimension,
                   'fitted_training_pairs': model.n_train_pairs,
                   'train': training, 'validation': validation}
            grid.append(row)
            if verbose:
                label = 'full' if rank is None else str(rank)
                print(f'  rank={label:>4} train MSE={training["mse"][0]:.6f} '
                      f'validation MSE={validation["mse"][0]:.6f} '
                      f'MAE={validation["mae"][0]:.6f} r={validation["pearson"][0]:.4f}', flush=True)
            score = validation['mse'][0]
            if score < best_score:
                best_model, best_score = model, score
    return best_model, grid


def load_checkpoint(path):
    """Load either factorized or unrestricted output from this rank experiment."""
    with np.load(path, allow_pickle=False) as saved:
        kind = str(saved['kind'])
        P, lam = int(saved['P']), float(saved['lam'])
        if kind == 'reduced_rank_ridge':
            model = ReducedRankVAR(P, int(saved['rank']), lam)
            model.input_factors = saved['input_factors'].copy()
            model.output_factors = saved['output_factors'].copy()
        elif kind == 'unrestricted_ridge':
            model = VAR(P, lam)
            model.coefficients = saved['coefficients'].copy()
        else:
            raise ValueError('Unknown checkpoint model kind')
        model.intercept = saved['intercept'].copy()
        model.n_train_pairs, model.training_start = int(saved['n_train_pairs']), int(saved['training_start'])
        preprocessing = {name: saved[name].copy() for name in ('mu', 'sd', 'log_transform')}
    N = len(model.intercept)
    arrays = [model.intercept, preprocessing['mu'], preprocessing['sd']]
    valid = (N > 0 and preprocessing['mu'].shape == (N,) and preprocessing['sd'].shape == (N,)
             and np.all(preprocessing['sd'] > 0) and model.training_start >= P)
    if isinstance(model, ReducedRankVAR):
        valid = valid and model.rank <= N and model.input_factors.shape == (P*N, model.rank) and model.output_factors.shape == (N, model.rank)
        arrays += [model.input_factors, model.output_factors]
    else:
        valid = valid and model.coefficients.shape == (P, N, N)
        arrays.append(model.coefficients)
    if not valid or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid checkpoint dimensions or values')
    return model, preprocessing


def run(dataset='cave_proofread', P=2, ranks=(16, 32, 64, 128, 256, 512),
        lambdas=(0.01, 0.1, 1), context=15, batch_size=256, output_dir=None, verbose=True):
    output = Path(output_dir or RUNS_DIR/f'rank_selection_{uuid4().hex[:8]}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty output directory')
    ranks, lambdas = list(ranks), list(lambdas)
    started = time.perf_counter()
    data = load_session(dataset)
    if verbose:
        print(f'Loaded {tuple(data.X.shape)} activity; {data.summary()["trials_per_split"]}', flush=True)
        print(f'VALIDATION SELECTION ONLY: P={P}, common context={context}, horizon=1', flush=True)
    model, grid = select_ranks(data.trials('train'), data.trials('val'), P, ranks,
                               lambdas, context, batch_size, verbose)
    selected = min(grid, key=lambda row: row['validation']['mse'][0])
    results = {'phase': 'validation_selection_only',
               'config': {'dataset': str(dataset), 'P': P, 'ranks': ranks, 'lambdas': lambdas,
                          'context': context, 'training_start': context, 'horizon': 1,
                          'batch_size': batch_size, 'include_unrestricted_reference': True},
               'selected': {k: selected[k] for k in ('P', 'rank', 'lambda', 'family',
                    'stored_parameter_count', 'coefficient_dimension', 'fitted_training_pairs')},
               'grid': grid, 'data': {**data.summary(),
                   'input_sha256': {name: sha256(data.directory/name) for name in ('X_features.npy', 'conditions.csv')},
                   'fit_trial_ids': np.flatnonzero(data.split=='train').tolist(),
                   'validation_trial_ids': np.flatnonzero(data.split=='val').tolist()},
               'metric_space': 'training-standardized log1p(clipped activity)',
               'method_reference': 'Mukherjee & Zhu (2011), doi:10.1002/sam.10138',
               'rank_definition': 'rank of stacked lag coefficients [P*N, N], excluding intercept',
               'parameter_count_convention': 'stored factor values plus intercept; coefficient_dimension subtracts r*r factor redundancy and is not statistical effective df',
               'source_sha256': {name: sha256(Path(__file__).parent/name) for name in
                    ('reduced_rank.py', 'select_ranks.py', 'var_p.py', 'lag_selection.py', 'data.py', 'evaluation.py')},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'elapsed_seconds': time.perf_counter()-started}
    serialized = json.dumps(results, indent=2, allow_nan=False)
    lines = ['VALIDATION SELECTION ONLY - no test forecasts computed',
             f'P={P}; common training/evaluation context={context}; one-step targets', '',
             'rank     lambda    stored values     train MSE       val MSE       val MAE    val Pearson']
    for row in grid:
        label = 'full' if row['rank'] is None else str(row['rank'])
        lines.append(f'{label:>4} {row["lambda"]:10g} {row["stored_parameter_count"]:16,d} '
                     f'{row["train"]["mse"][0]:13.9f} {row["validation"]["mse"][0]:13.9f} '
                     f'{row["validation"]["mae"][0]:13.9f} {row["validation"]["pearson"][0]:14.6f}')
    label = 'full' if selected['rank'] is None else selected['rank']
    lines += ['', f'Selected rank={label}, lambda={model.lam:g} by one-step validation MSE.',
              'Rank constraint and ridge strength learned/selected without test evaluation.']
    summary = '\n'.join(lines)+'\n'
    checkpoint = {'kind': selected['family'], 'P': model.P, 'lam': model.lam,
                  'intercept': model.intercept, 'n_train_pairs': model.n_train_pairs,
                  'training_start': model.training_start, 'mu': data.mu, 'sd': data.sd,
                  'log_transform': data.log_transform}
    if isinstance(model, ReducedRankVAR):
        checkpoint.update(rank=model.rank, input_factors=model.input_factors, output_factors=model.output_factors)
    else:
        checkpoint['coefficients'] = model.coefficients
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'model.npz', **checkpoint)
    (output/'selection.json').write_text(serialized+'\n')
    (output/'selection_summary.txt').write_text(summary)
    if verbose:
        print('\n'+summary, end='')
        print('Saved to', output, flush=True)
    return model, results, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--lags', type=int, default=2, dest='P', help='fixed activity lag order')
    parser.add_argument('--ranks', type=int, nargs='+', default=[16, 32, 64, 128, 256, 512])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.01, 0.1, 1])
    parser.add_argument('--context', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=Path)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f'Rank selection: {exc}\n')


if __name__ == '__main__':
    main()
