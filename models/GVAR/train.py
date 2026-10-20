"""Fit G-VAR for neural forecasting on a processed MICrONS session and evaluate it against copy-last.

    python models/GVAR/train.py --dataset cave_proofread                 # select P and K on validation
    python models/GVAR/train.py --dataset cave_all --P 3 --K 2           # fixed order
    python models/GVAR/train.py --P 15 --K 3 --leaky true                # + taps on S^k relu(x)
    python models/GVAR/train.py --P 15 --K 3 --video_pcs 32 --behaviour treadmill   # G-VAR-X, selects Q, lam_exo
    python models/GVAR/train.py --clamp false --intercept false          # the original G-VAR
    python models/GVAR/train.py --P 15 --K 3 --nl_layers 2 [--gated true]   # nonlinear G-VAR (gvar_nonlinear.py)
    python models/GVAR/train.py --P 15 --K 3 --strategy direct [--nl_layers 2]   # one head per horizon (gvar_direct.py)
    python models/GVAR/train.py --P 15 --K 3 --nl_layers 2 --nl_loss rollout     # trained through the recursive forecast
"""
# All choices (P, K, Q, lam_exo) minimise validation MSE averaged over h = 1..select_horizon.
import argparse
import copy
import itertools
import json
from pathlib import Path

import torch

from data import load_data, load_graph, shift_operator, trials
from exogenous import hold_unknown
from gvar import GVAR
from gvar_direct import DirectGVAR
import gvar_nonlinear

RUNS_DIR = Path(__file__).resolve().parent / 'runs'
P_GRID = (1, 2, 4, 6, 8, 10, 12, 15)
K_GRID = (0, 1, 2, 3)
Q_GRID = (1, 4, 8)
LAM_EXO_GRID = (0.01, 0.1, 1.0, 10.0)
CONTEXT = max(P_GRID)

DEFAULTS = dict(dataset='cave_proofread', graph='connectome', shift='mean', undirected=False,
                P=None, K=None, lam=1e-3, horizon=5, log_transform=True, val_frac=0.15, test_frac=0.15, seed=0,
                select_horizon=5, clamp=True, intercept=True, leaky=False, video_pcs=0, behaviour='', Q=None,
                lam_exo=None, nl_layers=0, nl_hidden=32, nl_K=None, gated=False, epochs=30, lr=1e-3,
                weight_decay=1e-4, batch_size=64, patience=3, device='auto', strategy='recursive', nl_loss='one_step',
                ss_epochs=0, finetune_linear=False)


def eval_windows(data, which, horizon):
    """All [CONTEXT + horizon, N] windows inside the trials of a split, stacked to [W, CONTEXT + horizon, N]."""
    # and the matching U windows [W, CONTEXT + horizon, E]
    def unfold(key):
        return torch.cat([x.unfold(0, CONTEXT + horizon, 1).permute(0, 2, 1) for x in trials(data, which, key)])
    return unfold('X'), unfold('U')


@torch.no_grad()
def evaluate(model, windows, horizon, known_ahead=()):
    """Forecast metrics per horizon for the model and two baselines (normalized).

    Exogenous inputs that are not known ahead (behaviour) are held at their value at the first target.
    """
    xw, uw = windows
    past, target = xw[:, :CONTEXT], xw[:, CONTEXT:]
    u = hold_unknown(uw, known_ahead, CONTEXT)[:, CONTEXT - model.Q + 1:] if model.n_exo else None
    dev = next(model.parameters()).device
    idx = torch.arange(len(past)).split(256 if dev.type == 'cuda' else 2048)  # small chunks fit a 4 GB GPU
    pred = torch.cat([model.rollout(past[i, -model.P:].to(dev), horizon, None if u is None else u[i].to(dev)).cpu()
                      for i in idx])

    def metrics(p):
        err = p - target
        pc, tc = p - p.mean(0), target - target.mean(0)
        r = (pc * tc).sum(0) / (pc.norm(dim=0) * tc.norm(dim=0) + 1e-8)  # [H, N]
        return {'mse': (err ** 2).mean((0, 2)).tolist(), 'mae': err.abs().mean((0, 2)).tolist(),
                'pearson': r.mean(-1).tolist(), 'mse_per_neuron_h1': (err[:, 0] ** 2).mean(0).tolist()}

    return {'gvar': metrics(pred),
            'copy_last': metrics(past[:, -1:].expand_as(target)),
            'mean': metrics(torch.zeros_like(target)),
            'n_windows': len(past)}


def build_shift(data, cfg):
    N = data['X'].shape[1]
    edge_index, edge_weight = load_graph(data['data_dir'], N, cfg['graph'], cfg['seed'])
    return shift_operator(edge_index, edge_weight, N, cfg['shift'], cfg['undirected'])


def default_run_name(cfg):
    name = f"{cfg['dataset']}_{cfg['graph']}"
    if not cfg['intercept']:
        name += '_noicpt'
    if not cfg['clamp']:
        name += '_noclamp'
    if cfg['leaky']:
        name += '_leaky'
    if cfg['nl_layers']:
        name += f"_nl{cfg['nl_layers']}" + ('_gated' if cfg['gated'] else '')
    if cfg['finetune_linear']:
        name += '_ft'
    if cfg['strategy'] == 'direct':
        name += '_direct'
    elif cfg['nl_loss'] != 'one_step' and (cfg['nl_layers'] or cfg['finetune_linear']):
        name += f"_{cfg['nl_loss']}" + (f"_ss{cfg['ss_epochs']}" if cfg['ss_epochs'] else '')
    if cfg['video_pcs']:
        name += f"_v{cfg['video_pcs']}"
    if cfg['behaviour']:
        name += '_' + cfg['behaviour'].replace(',', '+')
    return name


def run(cfg=None, data=None, run_name=None, verbose=True):
    """Fit G-VAR, test it and save the results."""
    cfg = {**DEFAULTS, **(cfg or {})}
    behaviour = tuple(b for b in cfg['behaviour'].split(',') if b)
    if data is None:
        data = load_data(cfg['dataset'], cfg['log_transform'], cfg['val_frac'], cfg['test_frac'],
                         cfg['video_pcs'], behaviour)
    E = data['U'].shape[1]
    S = build_shift(data, cfg)
    train_x, train_u = trials(data, 'train'), trials(data, 'train', 'U')
    val = eval_windows(data, 'val', cfg['select_horizon'])

    # validation MSE averaged over h = 1..select_horizon, and at h = 1
    def val_scores(model):
        mse = evaluate(model, val, cfg['select_horizon'], data['known_ahead'])['gvar']['mse']
        return sum(mse) / len(mse), mse[0]

    Ps = [cfg['P']] if cfg['P'] else P_GRID
    Ks = [cfg['K']] if cfg['K'] is not None else K_GRID
    Qs = [cfg['Q']] if cfg['Q'] else Q_GRID if E else [1]
    lam_exos = [cfg['lam_exo']] if cfg['lam_exo'] is not None else LAM_EXO_GRID if E else [0.0]
    grid, best = [], None
    for P, K, Q in itertools.product(Ps, Ks, Qs):
        if cfg['strategy'] == 'direct':
            model = DirectGVAR(S, P, K, max(cfg['horizon'], cfg['select_horizon']), cfg['leaky'], E, Q,
                               cfg['intercept'], data['known_ahead'])
        else:
            model = GVAR(S, P, K, cfg['leaky'], E, Q, cfg['intercept'])
        model.floor = data['floor'] if cfg['clamp'] else None
        st = model.statistics(train_x, train_u)  # penalties only change the solve
        for lam_exo in lam_exos:
            model.solve(st, cfg['lam'], lam_exo)
            val_mse, val_mse_h1 = val_scores(model)
            grid.append({'P': P, 'K': K, 'Q': Q, 'lam_exo': lam_exo, 'val_mse': val_mse, 'val_mse_h1': val_mse_h1})
            if best is None or val_mse < best[0]:
                best = (val_mse, copy.deepcopy(model), lam_exo)
    val_mse, model, lam_exo = best
    if verbose:
        exo = f"  Q={model.Q} lam_exo={lam_exo:g}  inputs {E}" if E else ''
        print(f"{cfg['dataset']} / {cfg['graph']}: P={model.P} K={model.K}{exo}  leaky={cfg['leaky']}  "
              f"clamp={cfg['clamp']}  intercept={model.c is not None}  parameters {model.parameter_count}  "
              f"val mse (h=1..{cfg['select_horizon']}) {val_mse:.4f}")

    linear, nonlinear = model, None
    if cfg['nl_layers'] or cfg['finetune_linear']:  # nl_layers 0 + finetune_linear: gradient-trained linear G-VAR
        dev = gvar_nonlinear.device(cfg['device'])
        model = gvar_nonlinear.NonlinearGVAR(copy.deepcopy(linear), cfg['nl_hidden'], cfg['nl_layers'], cfg['nl_K'],
                                             cfg['gated'], finetune_linear=cfg['finetune_linear']).to(dev)
        loss = 'direct' if cfg['strategy'] == 'direct' else cfg['nl_loss']
        if verbose:
            print(f"nonlinear G-VAR on {dev}: {cfg['nl_layers']} layers x {cfg['nl_hidden']}, gated={cfg['gated']}, "
                  f"{model.parameter_count} trainable parameters")

        def score(m):
            mse, mse_h1 = val_scores(m)
            return {'val_mse': mse, 'val_mse_h1': mse_h1}
        model, history, best_epoch = gvar_nonlinear.train(model, data, score, cfg['epochs'], cfg['lr'],
                                                          cfg['weight_decay'], cfg['batch_size'], cfg['patience'],
                                                          cfg['seed'], verbose, loss, cfg['select_horizon'],
                                                          cfg['ss_epochs'])
        nonlinear = {'parameter_count': model.parameter_count, 'best_epoch': best_epoch, 'history': history,
                     'device': str(dev)}

    test = evaluate(model, eval_windows(data, 'test', cfg['horizon']), cfg['horizon'], data['known_ahead'])
    if verbose:
        for name in ('gvar', 'copy_last', 'mean'):
            m = test[name]
            print(f"  test {name:10s} h=1 mse {m['mse'][0]:.4f} r {m['pearson'][0]:.4f}   "
                  f"h={cfg['horizon']} mse {m['mse'][-1]:.4f} r {m['pearson'][-1]:.4f}")

    # taps [K+1, P]; for direct, one such array per horizon
    taps = lambda f: [f(m) for m in linear.heads] if cfg['strategy'] == 'direct' else f(linear)
    results = {'config': cfg, 'P': linear.P, 'K': linear.K, 'Q': linear.Q, 'lam_exo': lam_exo,
               'parameter_count': linear.parameter_count, 'h': taps(lambda m: m.h.tolist()),
               'g': taps(lambda m: m.g.tolist()) if cfg['leaky'] else None, 'exo_names': data['exo_names'],
               'video_explained': data['video_explained'], 'grid': grid, 'nonlinear': nonlinear, 'test': test}
    run_dir = RUNS_DIR / (run_name or default_run_name(cfg))
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), run_dir / 'model.pt')
    return model, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_name', type=str, default=None)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f'--{k}', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=v)
        elif v is None:
            parser.add_argument(f'--{k}', type=float if k.startswith('lam') else int, default=v)
        else:
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args = vars(parser.parse_args())
    run_name = args.pop('run_name')
    run(args, run_name=run_name)
