"""Hub test on the output of influence.py.

1. Concentration: share of total per-neuron influence carried by the top 20% of neurons with outgoing edges,
   with a 95% interval from bootstrapping test trials. Compared with the same share for
   (1) out-degree and total synapse count, how concentrated the wiring is, and
   (2) the influence of the same model trained on a random graph (--null).
2. How hubs act: per hub, its outgoing edges and the share of its influence in its strongest edge.
3. Rich club: edges among the hubs compared with degree-preserving rewirings of the graph.
4. Does influence follow anatomy: Spearman correlation of edge influence with synapse count,
   and of neuron influence with out-degree and synapse count.

    python models/explainability/hubs.py --run gvar_cave_proofread_connectome --null gvar_cave_proofread_rewired
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'GVAR'))
from data import rewire  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / 'runs'
MEASURES = {'drop': 'drop_influence', 'sum': 'sum_edge_influence'}


def top_share(values, frac):
    """Share of the total carried by the top `frac` of entries along axis 0. Negative influence counts as zero."""
    v = np.sort(np.clip(values, 0, None), axis=0)[::-1]
    k = max(1, int(round(frac * v.shape[0])))
    return v[:k].sum(0) / np.maximum(v.sum(0), 1e-12)


def boot_weights(n_trials, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(n_trials, size=n_trials), minlength=n_trials) for _ in range(n_boot)])


def drop_share(run, frac, n_boot):
    """Top-share of drop influence, with a bootstrap interval over test trials."""
    z = np.load(RUNS_DIR / run / 'trial_sums.npz')
    w = boot_weights(len(z['counts']), n_boot)
    point = top_share(z['node_trial'].sum(1) / z['counts'].sum(), frac)
    boot = top_share((z['node_trial'] @ w.T) / (z['counts'] @ w.T), frac)
    return float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def rich_club(edges, hubs, num_nodes, n_rewire, seed=0):
    """Edges from hubs to hubs, against degree-preserving rewirings of the same graph."""
    ei = torch.tensor(edges[['src', 'dst']].values.T)
    is_hub = np.zeros(num_nodes, bool)
    is_hub[hubs] = True
    observed = int((is_hub[edges['src']] & is_hub[edges['dst']]).sum())
    null = []
    for r in range(n_rewire):
        s, d = rewire(ei, num_nodes, seed + r).numpy()
        null.append((is_hub[s] & is_hub[d]).sum())
    null = np.array(null)
    return {'hub_to_hub_edges': observed, 'rewired_mean': float(null.mean()),
            'ratio': float(observed / max(null.mean(), 1e-12)), 'p_value': float((1 + (null >= observed).sum()) / (1 + len(null)))}


def lorenz(values):
    v = np.sort(np.clip(values, 0, None))[::-1]
    return np.linspace(0, 1, len(v) + 1), np.concatenate([[0], np.cumsum(v) / max(v.sum(), 1e-12)])


def run(args):
    neurons = pd.read_csv(RUNS_DIR / args.run / 'neurons.csv')
    edges = pd.read_csv(RUNS_DIR / args.run / 'edges.csv')
    meta = json.loads((RUNS_DIR / args.run / 'meta.json').read_text())
    if not meta['complete']:
        print('Warning: influence.py ran on a subset of edges (--max_edges); the summed measure is incomplete.')
    num_nodes = meta['num_nodes']
    f = args.top
    infl = neurons[MEASURES[args.measure]].values
    k = max(1, int(round(f * len(neurons))))
    hub_rows = neurons.iloc[np.argsort(-infl)[:k]]

    # 1. Concentration
    concentration = {
        'influence_drop': drop_share(args.run, f, args.n_boot),
        'influence_sum_of_edges': float(top_share(neurons['sum_edge_influence'].values, f)),
        'wiring_out_degree': float(top_share(neurons['out_degree'].values.astype(float), f)),
        'wiring_out_synapses': float(top_share(neurons['out_synapses'].values, f)),
    }
    if args.null:
        concentration['null_influence_drop'] = drop_share(args.null, f, args.n_boot)
    lower = concentration['influence_drop'][1]
    # With --learned (Adaptive-GVAR) every neuron is a sender, most without anatomical edges, so the wiring shares are
    # near 100% and only the random-graph model is a fair comparison
    learned = meta.get('learned', False)
    compare = [] if learned else [concentration['wiring_out_degree'], concentration['wiring_out_synapses']]
    if args.null:
        compare.append(concentration['null_influence_drop'][2])
    if not compare:
        raise ValueError('--null is required for runs of influence.py --learned')
    verdict = bool(lower > max(compare))

    # 2. How hubs act
    strongest = edges.assign(pos=edges['influence'].clip(lower=0)).groupby('src')['pos'].agg(['max', 'sum'])
    neurons['top_edge_share'] = (strongest['max'] / strongest['sum'].replace(0, np.nan)).reindex(neurons['neuron']).values
    neurons['n_edges_ci_above_0'] = edges[edges['ci_low'] > 0].groupby('src').size().reindex(neurons['neuron']).fillna(0).values
    hub_profile = neurons.set_index('neuron').loc[hub_rows['neuron'], ['out_degree', 'out_synapses', MEASURES[args.measure],
                                                                     'n_edges_ci_above_0', 'top_edge_share', 'redundancy_gap']]

    # 3. Rich club and 4. anatomy
    club = rich_club(edges, hub_rows['neuron'].values, num_nodes, args.n_rewire)
    is_hub = edges['src'].isin(hub_rows['neuron']) & edges['dst'].isin(hub_rows['neuron'])
    club['mean_influence_hub_to_hub'] = float(edges.loc[is_hub, 'influence'].mean()) if is_hub.any() else None
    club['mean_influence_other'] = float(edges.loc[~is_hub, 'influence'].mean())
    rho = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    anatomy = {
        'edge_influence_vs_synapses': rho(edges['influence'], edges['synapses']),
        'neuron_influence_vs_out_degree': rho(infl, neurons['out_degree']),
        'neuron_influence_vs_out_synapses': rho(infl, neurons['out_synapses']),
        'edges_with_ci_above_0': int((edges['ci_low'] > 0).sum()),
        'edges_with_ci_below_0': int((edges['ci_high'] < 0).sum()),
        'n_edges': len(edges),
    }

    summary = {'run': args.run, 'null': args.null, 'top_fraction': f, 'n_senders': len(neurons), 'n_hubs': k,
               'hub_measure': MEASURES[args.measure], 'concentration': concentration,
               'evidence_for_hubs': verdict, 'rich_club': club, 'anatomy': anatomy,
               'hubs': hub_profile.reset_index().to_dict('records')}
    out = RUNS_DIR / args.run
    (out / 'hubs.json').write_text(json.dumps(summary, indent=2))
    neurons.to_csv(out / 'neurons.csv', index=False)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    curves = {'influence (drop)': neurons['drop_influence'], 'out-degree': neurons['out_degree'],
              'out synapses': neurons['out_synapses']}
    if args.null:
        curves['random-graph influence'] = pd.read_csv(RUNS_DIR / args.null / 'neurons.csv')['drop_influence']
    for name, v in curves.items():
        ax.plot(*lorenz(v.values.astype(float)), label=name)
    ax.plot([0, 1], [0, 1], color='grey', lw=0.8, ls='--', label='equal share')
    ax.axvline(f, color='grey', lw=0.8)
    senders = 'all neurons' if learned else 'neurons with outgoing edges'
    ax.set(xlabel=f'fraction of {senders} (most influential first)', ylabel='cumulative share',
           title=args.run)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'concentration.png', dpi=150)

    p, lo, hi = concentration['influence_drop']
    print(f"{args.run}: {len(neurons)} {senders}, top {f:.0%} = {k}")
    print(f"  top-{f:.0%} share  influence (drop) {p:.1%} [{lo:.1%}, {hi:.1%}]   sum of edges "
          f"{concentration['influence_sum_of_edges']:.1%}")
    print(f"                 out-degree {concentration['wiring_out_degree']:.1%}   out synapses "
          f"{concentration['wiring_out_synapses']:.1%}")
    if args.null:
        p, lo, hi = concentration['null_influence_drop']
        print(f"                 random graph {p:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  evidence for hubs: {verdict}")
    print(f"  rich club: {club['hub_to_hub_edges']} hub-to-hub edges vs {club['rewired_mean']:.1f} rewired "
          f"(ratio {club['ratio']:.2f}, p={club['p_value']:.3f})")
    print(f"  spearman edge influence vs synapses {anatomy['edge_influence_vs_synapses']:.2f}, neuron influence vs "
          f"out-degree {anatomy['neuron_influence_vs_out_degree']:.2f}")
    print(f'Saved {out / "hubs.json"} and {out / "concentration.png"}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='folder in models/explainability/runs')
    parser.add_argument('--null', default=None, help='the same model trained on a random graph')
    parser.add_argument('--top', type=float, default=0.2)
    parser.add_argument('--measure', choices=tuple(MEASURES), default='drop', help='per-neuron influence that defines hubs')
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--n_rewire', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
