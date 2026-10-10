"""Fit G-VAR for neural forecasting on a processed MICrONS session and evaluate it against copy-last.

    python models/GVAR/train.py --dataset cave_proofread                 # select P and K on validation
    python models/GVAR/train.py --dataset cave_all --P 3 --K 2           # fixed order
"""
import argparse
import itertools
import json
from pathlib import Path

import torch

from data import load_data, load_graph, shift_operator, trials
from gvar import GVAR

RUNS_DIR = Path(__file__).resolve().parent / 'runs'
P_GRID = (1, 2, 4, 6, 8, 10, 12, 15)
K_GRID = (0, 1, 2, 3)
CONTEXT = max(P_GRID)

DEFAULTS = dict(dataset='cave_proofread', graph='connectome', shift='mean', undirected=False,
                P=None, K=None, lam=1e-3, horizon=5, log_transform=True, val_frac=0.15, test_frac=0.15, seed=0)


def eval_windows(data, which, horizon):
    """All [CONTEXT + horizon, N] windows inside the trials of a split, stacked to [W, CONTEXT + horizon, N]."""
    return torch.cat([x.unfold(0, CONTEXT + horizon, 1).permute(0, 2, 1) for x in trials(data, which)])


@torch.no_grad()
def evaluate(model, windows, horizon):
    """Forecast metrics per horizon for the model and two baselines (normalized).
    """
    past, target = windows[:, :CONTEXT], windows[:, CONTEXT:]
    pred = torch.cat([model.rollout(b[:, -model.P:], horizon) for b in past.split(2048)])

    def metrics(p):
        err = p - target
        pc, tc = p - p.mean(0), target - target.mean(0)
        r = (pc * tc).sum(0) / (pc.norm(dim=0) * tc.norm(dim=0) + 1e-8)  # [H, N]
        return {'mse': (err ** 2).mean((0, 2)).tolist(), 'mae': err.abs().mean((0, 2)).tolist(),
                'pearson': r.mean(-1).tolist(), 'mse_per_neuron_h1': (err[:, 0] ** 2).mean(0).tolist()}

    return {'gvar': metrics(pred),
            'copy_last': metrics(past[:, -1:].expand_as(target)),
            'mean': metrics(torch.zeros_like(target)),
            'n_windows': len(windows)}


def build_shift(data, cfg):
    N = data['X'].shape[1]
    edge_index, edge_weight = load_graph(data['data_dir'], N, cfg['graph'], cfg['seed'])
    return shift_operator(edge_index, edge_weight, N, cfg['shift'], cfg['undirected'])


def run(cfg=None, data=None, run_name=None, verbose=True):
    """Fit G-VAR, test it and save the results."""
    cfg = {**DEFAULTS, **(cfg or {})}
    if data is None:
        data = load_data(cfg['dataset'], cfg['log_transform'], cfg['val_frac'], cfg['test_frac'])
    S = build_shift(data, cfg)
    train_trials = trials(data, 'train')
    val = eval_windows(data, 'val', 1)

    Ps = [cfg['P']] if cfg['P'] else P_GRID
    Ks = [cfg['K']] if cfg['K'] is not None else K_GRID
    grid, best = [], None
    for P, K in itertools.product(Ps, Ks):
        model = GVAR(S, P, K).fit(train_trials, cfg['lam'])
        val_mse = evaluate(model, val, 1)['gvar']['mse'][0]
        grid.append({'P': P, 'K': K, 'val_mse': val_mse})
        if best is None or val_mse < best[0]:
            best = (val_mse, model)
    model = best[1]

    test = evaluate(model, eval_windows(data, 'test', cfg['horizon']), cfg['horizon'])
    if verbose:
        print(f"{cfg['dataset']} / {cfg['graph']}: P={model.P} K={model.K}  val mse {best[0]:.4f}")
        for name in ('gvar', 'copy_last', 'mean'):
            m = test[name]
            print(f"  test {name:10s} h=1 mse {m['mse'][0]:.4f} r {m['pearson'][0]:.4f}   "
                  f"h={cfg['horizon']} mse {m['mse'][-1]:.4f} r {m['pearson'][-1]:.4f}")

    results = {'config': cfg, 'P': model.P, 'K': model.K, 'h': model.h.tolist(), 'grid': grid, 'test': test}
    run_dir = RUNS_DIR / (run_name or f"{cfg['dataset']}_{cfg['graph']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    return model, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_name', type=str, default=None)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f'--{k}', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=v)
        else:
            parser.add_argument(f'--{k}', type=type(v) if v is not None else int, default=v)
    args = vars(parser.parse_args())
    run_name = args.pop('run_name')
    run(args, run_name=run_name)
