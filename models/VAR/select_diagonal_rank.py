"""Validation-only diagonal-plus-low-rank VAR, with verified saved controls."""
import argparse
import copy
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
import scipy

try:
    from .compare_controls import evaluate_policies, load_checkpoint as load_control
    from .data import load_session
    from .diagonal_low_rank import DiagonalLowRankVAR, target_energy
    from .prediction_bounds import BoundedForecast, activity_floor
    from .train import sha256
    from .var_p import VAR
except ImportError:
    from compare_controls import evaluate_policies, load_checkpoint as load_control
    from data import load_session
    from diagonal_low_rank import DiagonalLowRankVAR, target_energy
    from prediction_bounds import BoundedForecast, activity_floor
    from train import sha256
    from var_p import VAR

RUNS_DIR = Path(__file__).resolve().parent/'runs'


def select_diagonal_rank(train_trials, validation_trials, floor, P=2, ranks=(16, 64),
                         population_lambdas=(0.01, 0.1), self_lambda=1e-4, context=15,
                         max_iter=20, patience=5, tol=1e-6, batch_size=256, verbose=True):
    train_trials, validation_trials = list(train_trials), list(validation_trials)
    ranks, population_lambdas = list(ranks), list(population_lambdas)
    if (not ranks or any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer))
                         or r < 1 for r in ranks)):
        raise ValueError('Provide positive integer ranks')
    if not population_lambdas or any(not np.isfinite(v) or v <= 0 for v in population_lambdas):
        raise ValueError('Provide finite positive population penalties')
    if (isinstance(patience, (bool, np.bool_)) or not isinstance(patience, (int, np.integer))
            or patience < 1):
        raise ValueError('Patience must be a positive integer')
    stats = VAR(P).statistics(train_trials, start=context)
    if max(ranks) > stats.num_neurons:
        raise ValueError('Rank exceeds neuron count')
    energy = target_energy(train_trials, stats)
    best_model, best_score, grid = None, float('inf'), []
    for population_lambda in population_lambdas:
        for rank in ranks:
            history, selected = [], {}

            def monitor(model, iteration, objective):
                metrics = evaluate_policies(model, validation_trials, floor, context, batch_size)
                score = metrics['bounded']['mse'][0]
                history.append({'iteration': iteration, 'training_objective': objective,
                                'validation_mse_unbounded': metrics['unbounded']['mse'][0],
                                'validation_mse_bounded': score})
                if score < selected.get('score', float('inf')):
                    selected.update(score=score, model=copy.deepcopy(model), metrics=metrics)
                if verbose:
                    print(f'  iteration={iteration:2d} train objective={objective:.6f} '
                          f'val(raw)={metrics["unbounded"]["mse"][0]:.6f} '
                          f'val(bounded)={score:.6f}', flush=True)
                return iteration-selected['model'].n_iterations < patience

            if verbose:
                print(f'Rank={rank}; self lambda={self_lambda:g}; population lambda={population_lambda:g}', flush=True)
            fitted = DiagonalLowRankVAR(P, rank, self_lambda, population_lambda).fit_statistics(
                stats, energy, max_iter=max_iter, tol=tol, on_iteration=monitor)
            model = selected['model']
            row = {'family': 'diagonal_low_rank', 'P': P, 'rank': int(rank),
                   'self_lambda': float(self_lambda), 'population_lambda': float(population_lambda),
                   'parameter_count': model.parameter_count, 'fitted_training_pairs': model.n_train_pairs,
                   'selected_iteration': model.n_iterations, 'iterations_run': fitted.n_iterations,
                   'stopping_reason': 'validation_patience' if fitted.stopping_reason == 'callback' else fitted.stopping_reason,
                   'population_component_frobenius_norm': float(np.sqrt(max(0.0,
                       np.sum((model.input_factors.T@model.input_factors)*(model.output_factors.T@model.output_factors))))),
                   'train': evaluate_policies(model, train_trials, floor, context, batch_size),
                   'validation': selected['metrics'], 'history': history}
            grid.append(row)
            if selected['score'] < best_score:
                best_model, best_score = model, selected['score']
    return best_model, grid


def verify_controls(directory, data, P, context, floor, batch_size=256):
    """Check provenance and replay validation; do not trust a mismatched run."""
    directory = Path(directory).resolve()
    saved = json.loads((directory/'selection.json').read_text())
    config = saved['config']
    if (saved['phase'] != 'validation_selection_only' or config['P'] != P
            or config['context'] != context or config['training_start'] != context or config['horizon'] != 1
            or config['selection_metric'] != 'bounded one-step validation MSE'):
        raise ValueError('Control run must match lag order, context and bounded one-step selection')
    for name in ('X_features.npy', 'conditions.csv'):
        if sha256(data.directory/name) != saved['data']['input_sha256'][name]:
            raise ValueError('Control run uses different processed inputs')
    for name, digest in saved['source_sha256'].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise ValueError(f'Control source changed: {name}; rerun controls before comparison')
    if (saved['data']['fit_trial_ids'] != np.flatnonzero(data.split == 'train').tolist()
            or saved['data']['validation_trial_ids'] != np.flatnonzero(data.split == 'val').tolist()):
        raise ValueError('Control split IDs differ')
    references, models = {}, {}
    for family in ('independent_ar', 'dense_var'):
        bounded, preprocessing, restored_family = load_control(directory/f'{family}.npz')
        if (family != restored_family or bounded.P != P or bounded.base.training_start != context
                or not np.array_equal(bounded.floor, floor)
                or not np.array_equal(preprocessing['mu'], data.mu)
                or not np.array_equal(preprocessing['sd'], data.sd)
                or bool(preprocessing['log_transform']) != data.log_transform):
            raise ValueError('Control checkpoint/preprocessing differs')
        expected = saved['selected'][family]
        expected_count = sum(max(0, len(x)-context) for x in data.trials('train'))
        if bounded.base.n_train_pairs != expected_count or bounded.base.lam != expected['lambda']:
            raise ValueError('Control fitting targets or selected penalty differ')
        metrics = evaluate_policies(bounded.base, data.trials('val'), floor, context, batch_size)
        for policy in ('unbounded', 'bounded'):
            for metric in ('mse', 'mae', 'pearson', 'r2'):
                if not np.allclose(metrics[policy][metric], expected['validation'][policy][metric], rtol=0, atol=1e-10):
                    raise ValueError('Control validation replay differs')
        references[family] = {**expected, 'validation': metrics,
                              'checkpoint_sha256': sha256(directory/f'{family}.npz')}
        models[family] = bounded.base
    return models, references


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as saved:
        model = DiagonalLowRankVAR(int(saved['P']), int(saved['rank']),
                                   float(saved['lam_self']), float(saved['lam_population']))
        for name in ('lag_weights', 'input_factors', 'output_factors', 'intercept'):
            setattr(model, name, saved[name].copy())
        model.n_train_pairs, model.training_start = int(saved['n_train_pairs']), int(saved['training_start'])
        model.n_iterations = int(saved['selected_iteration'])
        preprocessing = {name: saved[name].copy() for name in ('mu', 'sd', 'log_transform')}
        floor = saved['floor'].copy()
    N, P, rank = model.intercept.size, model.P, model.rank
    shapes = {'intercept': (N,), 'lag_weights': (P, N), 'input_factors': (P*N, rank), 'output_factors': (N, rank)}
    if (not N or rank > N or model.n_train_pairs < 1 or model.training_start < P or model.n_iterations < 0
            or any(getattr(model, name).shape != shape or not np.isfinite(getattr(model, name)).all()
                   for name, shape in shapes.items())):
        raise ValueError('Invalid diagonal-plus-low-rank checkpoint')
    expected = activity_floor(preprocessing['mu'], preprocessing['sd'], bool(preprocessing['log_transform']))
    if expected.shape != (N,) or not np.array_equal(expected, floor):
        raise ValueError('Checkpoint bound and preprocessing disagree')
    return BoundedForecast(model, floor), preprocessing


def run(controls_run, dataset='cave_proofread', P=2, ranks=(16, 64),
        population_lambdas=(0.01, 0.1), context=15, max_iter=20, patience=5,
        tol=1e-6, batch_size=256, output_dir=None, verbose=True):
    output = Path(output_dir or RUNS_DIR/f'diagonal_rank_{uuid4().hex[:8]}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty output directory')
    started = time.perf_counter()
    ranks, population_lambdas = list(ranks), list(population_lambdas)
    data = load_session(dataset)
    floor = activity_floor(data.mu, data.sd, data.log_transform)
    controls, references = verify_controls(controls_run, data, P, context, floor, batch_size)
    self_lambda = controls['independent_ar'].lam
    if verbose:
        print(f'Loaded {tuple(data.X.shape)} activity; {data.summary()["trials_per_split"]}', flush=True)
        print(f'VALIDATION ONLY: diagonal-plus-low-rank VAR({P}), context={context}, horizon=1', flush=True)
        print('Iteration 0 is the verified independent AR control. Population factors start at zero.', flush=True)
        print('Exact conditional updates; joint optimization is nonconvex. Validation selects the iterate.', flush=True)
    model, grid = select_diagonal_rank(data.trials('train'), data.trials('val'), floor, P, ranks,
                                       population_lambdas, self_lambda, context, max_iter,
                                       patience, tol, batch_size, verbose)
    selected = min(grid, key=lambda row: row['validation']['bounded']['mse'][0])
    winner = min([*references.values(), selected], key=lambda row: row['validation']['bounded']['mse'][0])
    result = {'phase': 'validation_selection_only',
              'config': {'dataset': str(dataset), 'P': P, 'ranks': ranks,
                         'population_lambdas': population_lambdas, 'self_lambda': self_lambda,
                         'context': context, 'training_start': context, 'horizon': 1,
                         'max_iter': max_iter, 'patience': patience, 'tol': tol, 'batch_size': batch_size,
                         'selection_metric': 'bounded one-step validation MSE',
                         'controls_run': str(Path(controls_run).resolve())},
              'selected': selected, 'best_overall_family': winner['family'], 'controls': references, 'grid': grid,
              'data': {**data.summary(),
                       'fit_trial_ids': np.flatnonzero(data.split == 'train').tolist(),
                       'validation_trial_ids': np.flatnonzero(data.split == 'val').tolist(),
                       'input_sha256': {name: sha256(data.directory/name) for name in ('X_features.npy', 'conditions.csv')}},
              'method': {'coefficient_structure': 'stacked diagonal self-history E(d) plus rank-limited L=A V.T; L may have diagonal entries',
                         'fit_objective': 'SSE/M + self_lambda*||d||^2 + population_lambda*||A V.T||_F^2; unpenalized intercept',
                         'optimization': 'alternating exact conditional solves, initialized at independent AR; no joint global-optimum guarantee',
                         'training_objective_history_units': 'fit objective divided by neuron count',
                         'parameter_count': 'stored self coefficients, factors and intercept; factor redundancy included',
                         'structural_inspiration': 'Wagenmaker et al., NeurIPS 2024, arXiv:2412.02529; our stacked rank and regularized conditional solver differ from their fitting procedure'},
              'source_sha256': {name: sha256(Path(__file__).parent/name) for name in
                               ('select_diagonal_rank.py', 'diagonal_low_rank.py', 'compare_controls.py',
                                'independent_ar.py', 'prediction_bounds.py', 'var_p.py', 'data.py',
                                'lag_selection.py', 'evaluation.py')},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'elapsed_seconds': time.perf_counter()-started}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    lines = ['VALIDATION ONLY - no test forecasts computed',
             f'P={P}; common context={context}; self lambda fixed at control value {self_lambda:g}', '',
             'family/rank       pop lambda   selected/ran   val(raw)    val(bounded)     MAE       Pearson']
    for family, ref in references.items():
        val = ref['validation']
        lines.append(f'{family:<18} {"-":>10} {"-":>14} {val["unbounded"]["mse"][0]:11.8f} '
                     f'{val["bounded"]["mse"][0]:14.9f} {val["bounded"]["mae"][0]:10.7f} {val["bounded"]["pearson"][0]:10.6f}')
    for row in grid:
        val = row['validation']
        steps = f'{row["selected_iteration"]}/{row["iterations_run"]}'
        lines.append(f'rank {row["rank"]:<13} {row["population_lambda"]:10g} {steps:>14} '
                     f'{val["unbounded"]["mse"][0]:11.8f} {val["bounded"]["mse"][0]:14.9f} '
                     f'{val["bounded"]["mae"][0]:10.7f} {val["bounded"]["pearson"][0]:10.6f}')
    lines += ['', f'Best overall family by bounded validation MSE: {winner["family"]}',
              f'Best added-model candidate: rank={model.rank}, population lambda={model.lam_population:g}, selected iteration={model.n_iterations}.',
              'Iteration 0 means the population addition did not improve the control.']
    summary = '\n'.join(lines)+'\n'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'model.npz', P=model.P, rank=model.rank,
                        lam_self=model.lam_self, lam_population=model.lam_population,
                        lag_weights=model.lag_weights, input_factors=model.input_factors,
                        output_factors=model.output_factors, intercept=model.intercept,
                        n_train_pairs=model.n_train_pairs, training_start=model.training_start,
                        selected_iteration=model.n_iterations, mu=data.mu, sd=data.sd,
                        log_transform=data.log_transform, floor=floor)
    (output/'selection.json').write_text(serialized+'\n')
    (output/'selection_summary.txt').write_text(summary)
    if verbose:
        print('\n'+summary, end='')
        print('Saved to', output, flush=True)
    return model, result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-run', type=Path, required=True)
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--lags', dest='P', type=int, default=2)
    parser.add_argument('--ranks', type=int, nargs='+', default=[16, 64])
    parser.add_argument('--population-lambdas', type=float, nargs='+', default=[0.01, 0.1])
    parser.add_argument('--context', type=int, default=15)
    parser.add_argument('--max-iter', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=Path)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        parser.exit(1, f'Diagonal-plus-low-rank VAR: {exc}\n')


if __name__ == '__main__':
    main()
