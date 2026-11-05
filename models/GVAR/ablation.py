# Graph ablation: each model with the connectome, a random graph (as many edges, shuffled weights) and no graph
# (S = 0, so every k >= 1 term vanishes), all at the connectome model's P = 15, K = 3.
# The seed sets the random graph and the network initialisation; closed-form fits on a fixed graph are deterministic.
#   python models/GVAR/ablation.py    fits or trains the missing runs, one at a time
import gc
import json

import pandas as pd
import torch

from data import load_data
from train import DEFAULTS, RUNS_DIR, default_run_name, is_saved, run

GRAPHS = ('connectome', 'random', 'none')
P6 = dict(epochs=30, patience=6)
MODELS = {  # name: (config, seeds)
    'G-VAR': (dict(), (0, 1, 2)),
    'direct G-VAR': (dict(strategy='direct'), (0, 1, 2)),
    'nonlinear G-VAR (direct)': (dict(strategy='direct', nl_layers=2, **P6), (0, 1, 2)),
    'nonlinear G-VAR (rollout)': (dict(nl_layers=2, nl_loss='rollout', **P6), (0, 1, 2)),
}


# (model, graph, seed, config, run name) for every run of the ablation
def configs():
    for model, (cfg, seeds) in MODELS.items():
        nonlinear = 'nl_layers' in cfg
        for graph in GRAPHS:
            for seed in seeds if nonlinear or graph == 'random' else (0,):
                c = {**cfg, 'graph': graph, 'seed': seed, 'P': 15, 'K': 3}  # what the connectome runs select
                yield model, graph, seed, c, default_run_name({**DEFAULTS, **c}) + ('_p6' if nonlinear else '')


def train_missing():
    data = None
    for model, graph, seed, cfg, name in configs():
        if not is_saved(cfg, name):
            print(f'--- {model} / {graph} / seed {seed}: {name}', flush=True)
            data = data or load_data('cave_proofread')
            run(cfg, data, name)
            gc.collect()
            torch.cuda.empty_cache()


# one row per saved run with its test metrics; missing (or stale) runs are skipped
def results():
    rows = []
    for model, graph, seed, cfg, name in configs():
        if not is_saved(cfg, name):
            continue
        with open(RUNS_DIR / name / 'results.json') as f:
            r = json.load(f)
        mse = r['test']['gvar']['mse']
        rows.append({'model': model, 'graph': graph, 'seed': seed, 'run': name, 'mse': mse, 'h1': mse[0],
                     'h5': mse[-1], 'avg': sum(mse) / len(mse), 'pearson h1': r['test']['gvar']['pearson'][0],
                     'per_neuron_h1': r['test']['gvar']['mse_per_neuron_h1'],
                     'best_epoch': (r['nonlinear'] or {}).get('best_epoch')})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    train_missing()
