"""Validation-only comparison of dense VAR and its independent-neuron control."""
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
    from .evaluation import ForecastMetrics
    from .independent_ar import IndependentAR
    from .lag_selection import history_batches
    from .prediction_bounds import BoundedForecast, activity_floor
    from .train import sha256
    from .var_p import VAR
except ImportError:
    from data import load_session
    from evaluation import ForecastMetrics
    from independent_ar import IndependentAR
    from lag_selection import history_batches
    from prediction_bounds import BoundedForecast, activity_floor
    from train import sha256
    from var_p import VAR

RUNS_DIR = Path(__file__).resolve().parent/'runs'


def evaluate_policies(model, trials, floor, context=15, batch_size=256):
    """One-step metrics for both policies from the SAME forecast origins.

    For multistep use BoundedForecast.rollout(), which bounds each feedback
    step. Bounding an already completed raw rollout is a different procedure.
    """
    bounded = BoundedForecast(model, floor)
    accumulators = {name: ForecastMetrics(1, len(model.intercept))
                    for name in ('unbounded', 'bounded')}
    for history, target in history_batches(trials, model.P, context, 1, batch_size):
        prediction = model.predict(history)[:, None, :]
        accumulators['unbounded'].update(prediction, target)
        accumulators['bounded'].update(np.maximum(prediction, bounded.floor), target)
    results = {}
    for name, metrics in accumulators.items():
        result = metrics.result()
        result['rmse'] = np.sqrt(result['mse']).tolist()
        result['r2'] = (1-metrics.squared_error/(metrics.target_ss+1e-8)).mean(1).tolist()
        result['n_windows'] = metrics.count
        results[name] = result
    return results


def select_controls(train_trials, validation_trials, floor, P=2, lambdas=(0.01, 0.1, 1),
                    context=15, batch_size=256, verbose=True):
    """Fit both families on common targets; select lambda using bounded val MSE."""
    train_trials, validation_trials = list(train_trials), list(validation_trials)
    lambdas = list(lambdas)
    base = VAR(P)
    if (isinstance(context, (bool, np.bool_)) or not isinstance(context, (int, np.integer))
            or context < P):
        raise ValueError('Context must be an integer >= P')
    if (isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1):
        raise ValueError('Batch size must be a positive integer')
    if not lambdas or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError('Provide finite positive ridge penalties')
    stats = base.statistics(train_trials, start=context)
    floor = np.asarray(floor, dtype=np.float64)
    if floor.shape != (stats.num_neurons,) or not np.isfinite(floor).all():
        raise ValueError('Provide one finite target lower bound per neuron')
    if verbose:
        print(f'Common fitting targets per model: {stats.count:,}', flush=True)
        print('Selecting lambda separately for each family by BOUNDED one-step validation MSE.', flush=True)
    models, grid, scores = {}, [], {}
    for lam in lambdas:
        for family, constructor in [('dense_var', VAR), ('independent_ar', IndependentAR)]:
            model = constructor(P, lam).fit_statistics(stats)
            training = evaluate_policies(model, train_trials, floor, context, batch_size)
            validation = evaluate_policies(model, validation_trials, floor, context, batch_size)
            row = {'family': family, 'P': int(P), 'lambda': float(lam),
                   'parameter_count': model.parameter_count, 'fitted_training_pairs': model.n_train_pairs,
                   'train': training, 'validation': validation}
            grid.append(row)
            score = validation['bounded']['mse'][0]
            if score < scores.get(family, float('inf')):
                models[family], scores[family] = model, score
            if verbose:
                print(f'{family:>14} lambda={lam:g} train(raw)={training["unbounded"]["mse"][0]:.6f} '
                      f'val(raw)={validation["unbounded"]["mse"][0]:.6f} val(bounded)={score:.6f} '
                      f'MAE={validation["bounded"]["mae"][0]:.6f} '
                      f'r={validation["bounded"]["pearson"][0]:.4f}', flush=True)
    return models, grid


def load_checkpoint(path):
    """Return the bounded forecast, preprocessing, and saved base model family."""
    with np.load(path, allow_pickle=False) as saved:
        family, P, lam = str(saved['family']), int(saved['P']), float(saved['lam'])
        if family == 'dense_var':
            model = VAR(P, lam)
            model.coefficients = saved['coefficients'].copy()
            weights = model.coefficients
        elif family == 'independent_ar':
            model = IndependentAR(P, lam)
            model.lag_weights = saved['lag_weights'].copy()
            weights = model.lag_weights
        else:
            raise ValueError('Unknown model family')
        model.intercept = saved['intercept'].copy()
        model.n_train_pairs, model.training_start = int(saved['n_train_pairs']), int(saved['training_start'])
        prep = {name: saved[name].copy() for name in ('mu', 'sd', 'log_transform')}
        floor = saved['floor'].copy()
    N = model.intercept.size
    expected = (P, N, N) if family == 'dense_var' else (P, N)
    if (model.intercept.shape != (N,) or not N or weights.shape != expected
            or not np.isfinite(weights).all() or not np.isfinite(model.intercept).all()
            or model.training_start < P or model.n_train_pairs < 1):
        raise ValueError('Invalid checkpoint shape or values')
    expected_floor = activity_floor(prep['mu'], prep['sd'], bool(prep['log_transform']))
    if expected_floor.shape != (N,) or not np.array_equal(floor, expected_floor):
        raise ValueError('Checkpoint lower bound and preprocessing disagree')
    return BoundedForecast(model, floor), prep, family


def run(dataset='cave_proofread', P=2, lambdas=(0.01, 0.1, 1), context=15,
        batch_size=256, output_dir=None, verbose=True):
    output = Path(output_dir or RUNS_DIR/f'controls_{uuid4().hex[:8]}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty output directory')
    started = time.perf_counter()
    lambdas = list(lambdas)
    data = load_session(dataset)
    floor = activity_floor(data.mu, data.sd, data.log_transform)
    if verbose:
        print(f'Loaded {tuple(data.X.shape)} activity; {data.summary()["trials_per_split"]}', flush=True)
        print(f'VALIDATION ONLY: dense VAR({P}) versus independent AR({P}); context={context}', flush=True)
    models, grid = select_controls(data.trials('train'), data.trials('val'), floor, P,
                                   lambdas, context, batch_size, verbose)
    selected = {family: min((row for row in grid if row['family'] == family),
                           key=lambda row: row['validation']['bounded']['mse'][0]) for family in models}
    result = {'phase': 'validation_selection_only',
              'config': {'dataset': str(dataset), 'P': P, 'lambdas': lambdas, 'context': context,
                         'training_start': context, 'horizon': 1, 'batch_size': batch_size,
                         'selection_metric': 'bounded one-step validation MSE',
                         'prediction_bound': 'raw zero mapped to standardized clipped-log1p units: -mu/sd'},
              'selected': selected, 'grid': grid,
              'data': {**data.summary(),
                       'fit_trial_ids': np.flatnonzero(data.split == 'train').tolist(),
                       'validation_trial_ids': np.flatnonzero(data.split == 'val').tolist(),
                       'input_sha256': {name: sha256(data.directory/name) for name in ('X_features.npy', 'conditions.csv')}},
              'metric_space': 'training-standardized log1p(clipped activity)',
              'fit_objective': 'sum squared error / number of target frames + lambda * sum squared lag coefficients; intercept unpenalized',
              'source_sha256': {name: sha256(Path(__file__).parent/name) for name in
                               ('compare_controls.py', 'independent_ar.py', 'prediction_bounds.py',
                                'var_p.py', 'lag_selection.py', 'evaluation.py', 'data.py')},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'elapsed_seconds': time.perf_counter()-started}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    lines = ['VALIDATION ONLY - no test forecasts computed',
             'Same fitting targets, split and bound policy for both families.',
             'Lambda selected separately by bounded one-step validation MSE.', '',
             'family             lambda    parameters   train(raw)     val(raw)  val(bounded)  bounded MAE  bounded r']
    for row in grid:
        tr, val = row['train'], row['validation']
        lines.append(f'{row["family"]:<18} {row["lambda"]:7g} {row["parameter_count"]:13,d} '
                     f'{tr["unbounded"]["mse"][0]:12.9f} {val["unbounded"]["mse"][0]:12.9f} '
                     f'{val["bounded"]["mse"][0]:13.9f} {val["bounded"]["mae"][0]:12.9f} '
                     f'{val["bounded"]["pearson"][0]:10.6f}')
    lines += ['', 'Selected penalties: '+', '.join(f'{family}={model.lam:g}' for family, model in models.items()),
              'Independent AR is a diagnostic control; dense VAR remains the population baseline.']
    summary = '\n'.join(lines)+'\n'
    output.mkdir(parents=True, exist_ok=True)
    for family, model in models.items():
        fields = dict(family=family, P=model.P, lam=model.lam, intercept=model.intercept,
                      n_train_pairs=model.n_train_pairs, training_start=model.training_start,
                      mu=data.mu, sd=data.sd, log_transform=data.log_transform, floor=floor)
        if family == 'dense_var':
            fields['coefficients'] = model.coefficients
        else:
            fields['lag_weights'] = model.lag_weights
        np.savez_compressed(output/f'{family}.npz', **fields)
    (output/'selection.json').write_text(serialized+'\n')
    (output/'selection_summary.txt').write_text(summary)
    if verbose:
        print('\n'+summary, end='')
        print('Saved to', output, flush=True)
    return models, result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--lags', type=int, dest='P', default=2)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.01, 0.1, 1])
    parser.add_argument('--context', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=Path)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f'VAR controls: {exc}\n')


if __name__ == '__main__':
    main()
