"""Evaluate frozen validation-selected VAR checkpoints on one-step test targets.

No fitting, refitting, hyperparameter search, or test-based model selection.
The primary policy is the bounded policy fixed during validation selection.
"""
import argparse
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
import scipy

try:
    from .compare_controls import evaluate_policies
    from .data import load_session
    from .evaluation import ForecastMetrics
    from .lag_selection import history_batches
    from .prediction_bounds import activity_floor
    from .select_diagonal_rank import load_checkpoint, verify_controls
    from .train import sha256
except ImportError:
    from compare_controls import evaluate_policies
    from data import load_session
    from evaluation import ForecastMetrics
    from lag_selection import history_batches
    from prediction_bounds import activity_floor
    from select_diagonal_rank import load_checkpoint, verify_controls
    from train import sha256

RUNS_DIR = Path(__file__).resolve().parent/'runs'


def metric_result(metrics):
    result = metrics.result()
    result['rmse'] = np.sqrt(result['mse']).tolist()
    result['r2'] = (1-metrics.squared_error/(metrics.target_ss+1e-8)).mean(1).tolist()
    result['n_windows'] = metrics.count
    return result


def score_frozen(models, test_trials, floor, context, batch_size=256):
    """One pass over common test origins; supplied models are never modified."""
    orders = {model.P for model in models.values()}
    if len(orders) != 1:
        raise ValueError('All frozen models must have the same lag order')
    P = orders.pop()
    N = len(floor)
    accumulators = {name: {policy: ForecastMetrics(1, N) for policy in ('unbounded', 'bounded')}
                    for name in models}
    naive = {name: ForecastMetrics(1, N) for name in ('copy_last', 'training_mean')}
    for history, target in history_batches(test_trials, P, context, 1, batch_size):
        for name, model in models.items():
            prediction = model.predict(history)[:, None, :]
            accumulators[name]['unbounded'].update(prediction, target)
            accumulators[name]['bounded'].update(np.maximum(prediction, floor), target)
        naive['copy_last'].update(history[:, -1:, :], target)
        naive['training_mean'].update(np.zeros_like(target), target)
    return {name: {policy: metric_result(metrics) for policy, metrics in policies.items()}
            for name, policies in accumulators.items()}, {name: metric_result(metrics) for name, metrics in naive.items()}


def prepare(run_dir, dataset='cave_proofread', controls_run=None, batch_size=256):
    """Verify provenance and frozen parameters before reading test predictions."""
    run_dir = Path(run_dir).resolve()
    selected_run = json.loads((run_dir/'selection.json').read_text())
    config, selected = selected_run['config'], selected_run['selected']
    if (selected_run['phase'] != 'validation_selection_only' or config['horizon'] != 1
            or config['training_start'] != config['context']
            or config['selection_metric'] != 'bounded one-step validation MSE'
            or selected['family'] != 'diagonal_low_rank'):
        raise ValueError('Expected the saved one-step validation-selected added-model run')
    for name, digest in selected_run['source_sha256'].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise ValueError(f'Source differs from the frozen run: {name}')
    data = load_session(dataset)
    for name, digest in selected_run['data']['input_sha256'].items():
        if sha256(data.directory/name) != digest:
            raise ValueError(f'Processed input differs from the selected run: {name}')
    for split, key in [('train', 'fit_trial_ids'), ('val', 'validation_trial_ids')]:
        if selected_run['data'][key] != np.flatnonzero(data.split == split).tolist():
            raise ValueError('Trial split differs from validation selection')
    floor = activity_floor(data.mu, data.sd, data.log_transform)
    controls_dir = Path(controls_run or config['controls_run']).resolve()
    controls, references = verify_controls(controls_dir, data, config['P'], config['context'], floor, batch_size)
    for family, reference in references.items():
        if reference['checkpoint_sha256'] != selected_run['controls'][family]['checkpoint_sha256']:
            raise ValueError('Control checkpoint differs from the selected-model comparison')
    bounded, preprocessing = load_checkpoint(run_dir/'model.npz')
    model = bounded.base
    same = (model.P == config['P'] == selected['P'] and model.rank == selected['rank']
            and model.lam_self == config['self_lambda'] == selected['self_lambda']
            and model.lam_population == selected['population_lambda']
            and model.n_iterations == selected['selected_iteration']
            and model.parameter_count == selected['parameter_count']
            and model.training_start == config['context']
            and model.n_train_pairs == selected['fitted_training_pairs']
            and model.n_train_pairs == controls['independent_ar'].n_train_pairs
            and np.array_equal(bounded.floor, floor)
            and np.array_equal(preprocessing['mu'], data.mu)
            and np.array_equal(preprocessing['sd'], data.sd)
            and bool(preprocessing['log_transform']) == data.log_transform)
    if not same:
        raise ValueError('Selected checkpoint parameters/preprocessing differ from its validation record')
    replay = evaluate_policies(model, data.trials('val'), floor, config['context'], batch_size)
    for policy in ('unbounded', 'bounded'):
        for metric in ('mse', 'mae', 'pearson', 'r2'):
            if not np.allclose(replay[policy][metric], selected['validation'][policy][metric], rtol=0, atol=1e-10):
                raise ValueError('Selected checkpoint validation replay differs')
    models = {'dense_var': controls['dense_var'], 'independent_ar': controls['independent_ar'],
              'diagonal_low_rank': model}
    return data, models, floor, selected_run, controls_dir


def run(run_dir, dataset='cave_proofread', controls_run=None, batch_size=256,
        output_dir=None, verbose=True):
    output = Path(output_dir or RUNS_DIR/f'test_one_step_{uuid4().hex[:8]}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty test output directory')
    if (isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1):
        raise ValueError('Batch size must be a positive integer')
    started = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    data, models, floor, validation, controls_dir = prepare(run_dir, dataset, controls_run, batch_size)
    context = validation['config']['context']
    test_ids = np.flatnonzero(data.split == 'test').tolist()
    count = sum(max(0, len(x)-context) for x in data.trials('test'))
    if not count:
        raise ValueError('No test windows for the frozen context')
    checkpoints = {'diagonal_low_rank': sha256(run_dir/'model.npz'),
                   **{name: sha256(controls_dir/f'{name}.npz') for name in ('dense_var', 'independent_ar')}}
    # Write this ledger BEFORE any test forecast is computed, making the
    # preselected configuration and prediction policy inspectable.
    frozen = {'phase': 'frozen_one_step_test_protocol', 'selected_run_id': run_dir.name,
              'controls_run_id': controls_dir.name, 'dataset': str(dataset),
              'context': context, 'horizon': 1, 'primary_prediction_policy': 'bounded',
              'secondary_policy': 'unbounded diagnostic, not used to select models',
              'selected_configuration': {key: validation['selected'][key] for key in
                  ('P', 'rank', 'self_lambda', 'population_lambda', 'selected_iteration', 'parameter_count')},
              'control_lambdas': {name: models[name].lam for name in ('dense_var', 'independent_ar')},
              'training_pairs_per_model': {name: model.n_train_pairs for name, model in models.items()},
              'test_trial_ids': test_ids, 'expected_test_windows': count,
              'checkpoint_sha256': checkpoints, 'input_sha256': validation['data']['input_sha256'],
              'evaluation_source_sha256': sha256(Path(__file__)),
              'model_source_sha256': validation['source_sha256'],
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'refit': False, 'test_based_selection': False}
    output.mkdir(parents=True, exist_ok=True)
    (output/'frozen_protocol.json').write_text(json.dumps(frozen, indent=2, allow_nan=False)+'\n')
    if verbose:
        print(f'FROZEN TEST EVALUATION: {len(test_ids)} trials, {count:,} common one-step windows', flush=True)
        print('Saved checkpoints only; bounded primary predictions; no fitting or parameter selection.', flush=True)
    metrics, baselines = score_frozen(models, data.trials('test'), floor, context, batch_size)
    if any(metric['n_windows'] != count for policies in metrics.values() for metric in policies.values()):
        raise RuntimeError('Unexpected test window count')
    validation_metrics = {name: validation['controls'][name]['validation'] for name in ('dense_var', 'independent_ar')}
    validation_metrics['diagonal_low_rank'] = validation['selected']['validation']
    result = {'phase': 'frozen_test_evaluation', 'protocol': frozen,
              'metric_space': 'training-standardized log1p(clipped activity)',
              'models': {name: {'parameter_count': models[name].parameter_count,
                                 'test': policies, 'validation': validation_metrics[name]}
                         for name, policies in metrics.items()},
              'baselines': baselines,
              'scope': 'One session, later trials; overlapping windows; settings fixed by prior validation. No uncertainty interval.',
              'elapsed_seconds': time.perf_counter()-started}
    lines = ['FROZEN ONE-STEP TEST RESULTS - settings selected on validation, no refitting',
             f'{len(test_ids)} test trials; {count:,} identical windows; context={context}; primary policy=bounded', '',
             'model                 parameters     test MSE    test RMSE     test MAE    Pearson r      macro R2']
    for name, row in result['models'].items():
        m = row['test']['bounded']
        lines.append(f'{name:<22} {row["parameter_count"]:10,d} {m["mse"][0]:12.9f} '
                     f'{m["rmse"][0]:12.9f} {m["mae"][0]:12.9f} {m["pearson"][0]:12.6f} {m["r2"][0]:12.6f}')
    for name, m in baselines.items():
        lines.append(f'{name:<22} {"-":>10} {m["mse"][0]:12.9f} {m["rmse"][0]:12.9f} '
                     f'{m["mae"][0]:12.9f} {m["pearson"][0]:12.6f} {m["r2"][0]:12.6f}')
    summary = '\n'.join(lines)+'\n'
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'metrics_summary.txt').write_text(summary)
    if verbose:
        print('\n'+summary, end='')
        print('Saved to', output, flush=True)
    return result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--controls-run', type=Path)
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=Path)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f'Frozen test evaluation: {exc}\n')


if __name__ == '__main__':
    main()
