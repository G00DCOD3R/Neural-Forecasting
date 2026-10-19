# Select a VAR-X on validation, evaluate it on test and save the run, like train.py does for VAR1.
#   ./.venv/bin/python models/VAR/train_varx.py --dataset cave_proofread --video-pcs 32 --behaviour treadmill
# Penalties and Q minimise the validation MSE averaged over horizons 1..select_horizon.
import argparse
import itertools
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

try:
    from .data import load_session
    from .evaluation import ForecastMetrics
    from .train import sha256
    from .varx import VARX
except ImportError:
    from data import load_session
    from evaluation import ForecastMetrics
    from train import sha256
    from varx import VARX
from exogenous import hold_unknown  # repository root, put on the path by data.py

RUNS_DIR = Path(__file__).resolve().parent / 'runs'


# Windows within trials: last observation [b, N], targets [b, horizon, N], exogenous rows [b, Q-1+horizon, E].
# Inputs not known ahead are held from the first target on.
def forecast_batches_exo(data, which, Q, context=15, horizon=5, batch_size=256):
    if Q - 1 > context:
        raise ValueError('Q - 1 must not exceed the context')
    for x, u in zip(data.trials(which), data.exo_trials(which)):
        for start in range(context, len(x) - horizon + 1, batch_size):
            origins = np.arange(start, min(start + batch_size, len(x) - horizon + 1))
            uw = u[origins[:, None] + np.arange(-Q + 1, horizon)]
            yield x[origins - 1], x[origins[:, None] + np.arange(horizon)], hold_unknown(uw, data.known_ahead, Q - 1)


# VAR-X, copy-last and training-mean metrics on identical windows
def evaluate_varx(model, data, which, context=15, horizon=5, batch_size=256, floor=None):
    N = data.X.shape[1]
    metrics = {name: ForecastMetrics(horizon, N) for name in ('varx', 'copy_last', 'mean')}
    for previous, target, u in forecast_batches_exo(data, which, model.Q, context, horizon, batch_size):
        metrics['varx'].update(model.rollout(previous, u, horizon, floor), target)
        metrics['copy_last'].update(np.broadcast_to(previous[:, None], target.shape), target)
        metrics['mean'].update(np.zeros_like(target), target)
    return {**{name: m.result() for name, m in metrics.items()}, 'n_windows': metrics['varx'].count,
            'context': context, 'horizon': horizon}


# Grid over Q, lam and lam_exo; statistics are computed once per Q
def select_varx(data, lambdas, lam_exos, Qs, context=15, select_horizon=5, floor=None, verbose=True):
    grid, best = [], None
    for Q in Qs:
        model = VARX(Q)
        st = model.statistics(data.trials('train'), data.exo_trials('train'))
        for lam, lam_exo in itertools.product(lambdas, lam_exos):
            model.solve(st, lam, lam_exo)
            mse = evaluate_varx(model, data, 'val', context, select_horizon, floor=floor)['varx']['mse']
            grid.append({'Q': Q, 'lam': lam, 'lam_exo': lam_exo, 'val_mse': float(np.mean(mse)),
                         'val_mse_h1': mse[0]})
            if verbose:
                print(f"  Q={Q} lam={lam:g} lam_exo={lam_exo:g}  val mse {grid[-1]['val_mse']:.4f}", flush=True)
            if best is None or grid[-1]['val_mse'] < best[0]:
                best = (grid[-1]['val_mse'], (Q, lam, lam_exo))
    Q, lam, lam_exo = best[1]
    return VARX(Q).fit(data.trials('train'), data.exo_trials('train'), lam, lam_exo), grid


def run(dataset='cave_proofread', output_dir=None, video_pcs=0, behaviour=(), lambdas=(0.01, 0.1, 1.0),
        lam_exos=(0.1, 1.0, 10.0), Qs=(1, 4, 8), context=15, horizon=5, select_horizon=5, clamp=False,
        verbose=True):
    output = Path(output_dir or RUNS_DIR / f'varx_{uuid4().hex[:8]}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'{output} is not an empty directory; choose another output path')
    started = time.perf_counter()
    data = load_session(dataset, video_pcs=video_pcs, behaviour=behaviour)
    E = data.U.shape[1]
    if not E:  # nothing to lag or penalise
        Qs, lam_exos = (1,), (1.0,)
    floor = data.X[data.split[data.trial_id] == 'train'].min(0) if clamp else None  # lowest training value
    if verbose:
        print(f'VAR-X on {tuple(data.X.shape)}, {E} exogenous inputs {data.exo_names[:3]}...', flush=True)
    model, grid = select_varx(data, lambdas, lam_exos, Qs, context, select_horizon, floor, verbose)
    test = evaluate_varx(model, data, 'test', context, horizon, floor=floor)
    results = {
        'config': {'dataset': str(dataset), 'video_pcs': video_pcs, 'behaviour': list(behaviour),
                   'lambdas': list(lambdas), 'lam_exos': list(lam_exos), 'Qs': list(Qs), 'context': context,
                   'horizon': horizon, 'select_horizon': select_horizon, 'clamp': clamp},
        'Q': model.Q, 'lam': model.lam, 'lam_exo': model.lam_exo, 'parameter_count': model.parameter_count,
        'n_train_pairs': model.n_train_pairs, 'exo_names': data.exo_names, 'grid': grid, 'test': test,
        'data': {**data.summary(),
                 'sha256': {name: sha256(data.directory / name) for name in ('X_features.npy', 'conditions.csv')}},
        'elapsed_seconds': time.perf_counter() - started,
    }
    serialized = json.dumps(results, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'model.npz', Q=model.Q, lam=model.lam, lam_exo=model.lam_exo,
                        coefficients=model.coefficients, exo_coefficients=model.exo_coefficients,
                        intercept=model.intercept, mu=data.mu, sd=data.sd)
    (output / 'results.json').write_text(serialized + '\n', encoding='utf-8')
    if verbose:
        m = test['varx']
        print(f"Selected Q={model.Q} lam={model.lam:g} lam_exo={model.lam_exo:g}; test h=1 mse {m['mse'][0]:.4f}, "
              f"h={horizon} mse {m['mse'][-1]:.4f}\nSaved to {output}", flush=True)
    return model, results, output


def main():
    parser = argparse.ArgumentParser(description='Select, test and save a VAR-X')
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--video-pcs', type=int, default=0)
    parser.add_argument('--behaviour', nargs='*', default=[])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.01, 0.1, 1.0])
    parser.add_argument('--lam-exos', type=float, nargs='+', default=[0.1, 1.0, 10.0])
    parser.add_argument('--Qs', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--select-horizon', type=int, default=5)
    parser.add_argument('--clamp', action='store_true')
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.video_pcs, tuple(args.behaviour), args.lambdas, args.lam_exos,
        args.Qs, horizon=args.horizon, select_horizon=args.select_horizon, clamp=args.clamp)


if __name__ == '__main__':
    main()
