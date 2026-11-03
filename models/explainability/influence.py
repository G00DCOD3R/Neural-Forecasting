"""Per-edge and per-neuron influence of a trained graph forecaster, measured by removing edges at test time.

Per-edge influence of j -> i: the rise in neuron i's one-step test MSE when that edge is removed.
Per-neuron influence of j, two ways:
    (a) sum of the per-edge influences of j's outgoing edges
    (b) remove all of j's outgoing edges at once and sum the rise in MSE over all neurons
MSE is in normalised units (log1p, z-scored on training data), on every 15-frame test window inside a trial.
95% intervals come from bootstrapping test trials.

Train the model first, then point this script at its run:

    python models/GVAR/train.py --dataset cave_proofread --horizon 1
    python models/explainability/influence.py --model gvar --run cave_proofread_connectome

    python models/stgnn/train.py --model dcrnn --dataset cave_proofread
    python models/explainability/influence.py --model dcrnn --run cave_proofread_dcrnn_connectome --stride 4

    python models/explainability/influence.py --model adaptive_gvar --run adaptive_gvar_sparsemax --stride 4
    python models/explainability/influence.py --model adaptive_gvar --run adaptive_gvar_sparsemax --stride 4 --learned
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

MODELS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS_DIR / 'GVAR'))
sys.path.insert(0, str(MODELS_DIR))
from common import load_session  # noqa: E402
from data import DATASETS, load_graph, shift_operator  # noqa: E402
from gvar import GVAR  # noqa: E402
from temporal_adaptive_gvar import AdaptiveGVAR  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / 'runs'
HISTORY = 15  # same windows as G-VAR's evaluation and the graph baselines


def test_windows(data, history):
    """[W, history + 1, N] activity windows inside test trials, the matching exogenous windows, and each window's trial."""
    xs, us, trial_of = [], [], []
    for trial in np.flatnonzero(data['split'] == 'test'):
        mask = data['trial_id'] == trial
        x = data['X'][mask]
        if x.shape[0] <= history:
            continue
        w = x.unfold(0, history + 1, 1).permute(0, 2, 1)
        xs.append(w)
        if data['U'] is not None:
            us.append(data['U'][mask].unfold(0, history + 1, 1).permute(0, 2, 1))
        trial_of.append(np.full(len(w), trial))
    return torch.cat(xs), (torch.cat(us) if us else None), np.concatenate(trial_of)


class GVARPredictor:
    """G-VAR is linear, so neuron t's forecast is sum_p C[t, p] . x_{t-p} with C[t, p] = sum_k h_kp (S^k)[t].

    Only the rows of the targets we need are computed, which makes thousands of ablations cheap.
    """

    def __init__(self, run_dir, data, past):
        res = json.loads((run_dir / 'results.json').read_text())
        self.cfg, self.P, self.K = res['config'], res['P'], res['K']
        self.h = torch.tensor(res['h'], dtype=torch.float32)
        self.N = data['X'].shape[1]
        self.edge_index, self.edge_weight = load_graph(data['data_dir'], self.N, self.cfg['graph'], self.cfg['seed'])
        self.lags = past[:, -self.P:].flip(1)  # [W, P, N], index p-1 = lag p
        self.full = self.shift(torch.ones(self.edge_index.shape[1], dtype=torch.bool))
        self.full_dense = self.full.to_dense()
        self.reach = (self.full_dense != 0).float()  # reach[t, m] = 1 if m feeds t

        # The row formula must reproduce the model's own forecasts
        model = GVAR(self.full, self.P, self.K)
        model.h.data.copy_(self.h)
        check = past[:256, -self.P:]
        assert torch.allclose(model(check), self.predict(None, torch.arange(self.N))[:256], atol=1e-4)

    def shift(self, keep):
        return shift_operator(self.edge_index[:, keep], self.edge_weight[keep], self.N,
                              self.cfg['shift'], self.cfg['undirected'])

    @torch.no_grad()
    def predict(self, keep, targets):
        """Forecasts [W, len(targets)] with only the edges in `keep` (None = all edges)."""
        S = self.full if keep is None else self.shift(keep)
        R = torch.zeros(len(targets), self.N)
        R[torch.arange(len(targets)), targets] = 1
        C = self.h[0][None, :, None] * R[:, None]  # [T, P, N]
        for k in range(1, self.K + 1):
            R = torch.sparse.mm(S.t(), R.T).T  # rows of S^k
            C = C + self.h[k][None, :, None] * R[:, None]
        W = self.lags.shape[0]
        return self.lags.reshape(W, -1) @ C.reshape(len(targets), -1).T

    def affected(self, keep):
        """Neurons whose forecast can change: rows of S that changed, plus up to K-1 hops downstream of them."""
        changed = (self.shift(keep).to_dense() != self.full_dense).any(1).float()
        hit = changed.clone()
        for _ in range(self.K - 1):
            hit = ((self.reach @ hit) + hit).clamp(max=1)
        return torch.nonzero(hit).flatten()


class GNNPredictor:
    """DCRNN or Graph WaveNet from models/stgnn, run for every ablation.

    Both models only mix neurons through the graph, so removing an edge cannot change neurons outside its
    connected component. Each ablation runs on the components of the targets only, which gives the same
    forecasts (unless Graph WaveNet learned its own adjacency, which links every pair of neurons).
    """

    def __init__(self, run_dir, data, past, exog, device, batch_size):
        spec = importlib.util.spec_from_file_location('stgnn_train', MODELS_DIR / 'stgnn' / 'train.py')
        stgnn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stgnn)
        res = json.loads((run_dir / 'results.json').read_text())
        self.cfg = res['config']
        self.N = data['X'].shape[1]
        if res['learned_adjacency']:
            print('Warning: this Graph WaveNet also learned its own adjacency, which edge removal does not touch. '
                  'Train with --no_learned_adjacency for this analysis.')
        self.model = stgnn.build_model(
            self.cfg['model'], self.N, self.cfg['horizon'], 0 if exog is None else exog.shape[-1],
            self.cfg['hidden'], self.cfg['ff_size'], self.cfg['layers'], res['learned_adjacency'], self.cfg['dropout'],
        )
        self.model.load_state_dict(torch.load(run_dir / 'best.pt', map_location='cpu'))
        self.model.to(device).eval()
        # DCRNN is built with cache_support=True, which would keep using the first graph it saw
        self.model.cache_support = False
        self.edge_index, self.edge_weight = load_graph(data['data_dir'], self.N, self.cfg['graph'], self.cfg['seed'])
        self.edge_weight = self.edge_weight.float()
        self.stgnn, self.past, self.exog = stgnn, past[:, -self.cfg['history']:].to(device), exog
        self.device, self.batch_size = device, batch_size
        if res['learned_adjacency']:
            self.component = torch.zeros(self.N, dtype=torch.long)
        else:
            adj = sp.coo_matrix((np.ones(self.edge_index.shape[1]), self.edge_index.numpy()), shape=(self.N, self.N))
            self.component = torch.from_numpy(connected_components(adj, directed=False)[1])

    def nodes_for(self, targets):
        """Neurons in the same connected components as `targets`, sorted."""
        return torch.nonzero(torch.isin(self.component, self.component[targets])).flatten()

    @torch.no_grad()
    def predict(self, keep, targets):
        keep = torch.ones(self.edge_index.shape[1], dtype=torch.bool) if keep is None else keep.clone()
        nodes = self.nodes_for(targets)
        keep &= torch.isin(self.edge_index[0], nodes)  # same component, so the target is in `nodes` too
        relabel = torch.full((self.N,), -1, dtype=torch.long)
        relabel[nodes] = torch.arange(len(nodes))
        ei, ew = relabel[self.edge_index[:, keep]].to(self.device), self.edge_weight[keep].to(self.device)
        pick = relabel[targets]
        for m in self.model.modules():
            if hasattr(m, '_support'):
                m._support = None
        out = []
        for i in range(0, len(self.past), self.batch_size):
            u = None if self.exog is None else self.exog[i:i + self.batch_size].to(self.device)
            past = self.past[i:i + self.batch_size, :, nodes.to(self.device)]
            pred = self.stgnn.predict(self.model, past, u, ei, ew)
            out.append(pred[:, 0, pick].cpu())
        return torch.cat(out)

    def affected(self, keep):
        # message passing over 15 steps can reach the whole component
        return self.nodes_for(self.edge_index[0, ~keep])


def adaptive_checkpoint(run):
    """models/GVAR/checkpoints/<run>/best.pt, from adaptive_gvar.ipynb or models/GVAR/train_adaptive.py."""
    path = MODELS_DIR / 'GVAR' / 'checkpoints' / run / 'best.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('model_type') != 'temporal_adaptive_gvar_v1':
        raise ValueError(f'{path} is not an activity-dependent Adaptive-GVAR checkpoint')
    return path, saved


def check_same_graph(saved_S, A):
    """The checkpoint may list neurons in another order (the model has no per-neuron parameters, so it gives the same
    forecasts either way), but it must have been trained on the same graph: same weights and degree sequences."""
    if torch.equal(saved_S, A):
        return
    signature = lambda M: (M[M != 0].sort().values, (M != 0).sum(0).sort().values, (M != 0).sum(1).sort().values)  # noqa: E731
    for a, b in zip(signature(saved_S), signature(A)):
        if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=1e-6):
            raise ValueError('The checkpoint was trained on a different graph than the one loaded for this dataset')


class AdaptiveGVARPredictor:
    """Activity-dependent Adaptive-GVAR (models/GVAR/temporal_adaptive_gvar.py).

    Each window w is forecast with S_w = (1 - alpha) A + alpha C_w: A is the row-normalised anatomical graph and C_w
    a graph the model infers from that window's activity. C_w does not depend on A, so removing anatomical edges
    leaves it unchanged. It is computed once for all test windows and stored as one block-diagonal sparse matrix,
    and each ablation only redoes the K graph hops.

    learned=True also removes neuron j as a source of C_w in the per-neuron ablation. Its score is masked before the
    sparsemax (or softmax), so each row of C_w still sums to one. Only rows where j had a nonzero weight change.
    """

    def __init__(self, run, data, past, device, learned, chunk=8):
        path, saved = adaptive_checkpoint(run)
        self.cfg = {**saved['config'], 'graph': saved.get('graph', 'connectome'),
                    'graph_seed': saved.get('graph_seed', 0), 'checkpoint': str(path.relative_to(MODELS_DIR))}
        self.N, self.device, self.learned = data['X'].shape[1], device, learned
        self.edge_index, self.edge_weight = load_graph(data['data_dir'], self.N, self.cfg['graph'], self.cfg['graph_seed'])
        A = self.shift(torch.ones(self.edge_index.shape[1], dtype=torch.bool))
        check_same_graph(saved['state_dict']['S'], A.to_dense())
        self.model = AdaptiveGVAR(A, **saved['config'])
        self.model.load_state_dict(dict(saved['state_dict'], S=A.to_dense()))
        self.model.to(device).eval()
        self.A = A.to(device)
        self.P, self.K, self.alpha = self.model.P, self.model.K, self.model.alpha
        self.h = self.model.h.detach()
        x = past[:, -self.P:]
        self.x0 = x.flip(1).transpose(1, 2).contiguous().to(device)  # [W, N, P], lag 1 first
        self.learner = self.model.graph_learner

        # Learned graphs of all test windows, block-diagonal [W*N, W*N]; keep Q and K to redo rows without a source
        W, N = len(x), self.N
        Qs, Ks, crow, col, col_local, val = [], [], [torch.zeros(1, dtype=torch.long, device=device)], [], [], []
        nnz = 0
        with torch.no_grad():
            for s in range(0, W, chunk):
                xb = x[s:s + chunk].to(device)
                b = len(xb)
                _, hidden = self.learner.encoder(xb.transpose(1, 2).reshape(b * N, self.P, 1))
                H = hidden[0].reshape(b, N, -1)
                Q, K = self.learner.W_q(H), self.learner.W_k(H)
                C = self.learner.normalize_scores(Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1]))
                if s == 0:
                    assert torch.allclose(C, self.learner(xb), atol=1e-6)
                Qs.append(Q)
                Ks.append(K)
                Cs = C.reshape(b * N, N).to_sparse_csr()
                window = torch.arange(s, s + b, device=device).repeat_interleave(N).repeat_interleave(Cs.crow_indices().diff())
                crow.append(Cs.crow_indices()[1:] + nnz)
                col.append(Cs.col_indices() + window * N)
                col_local.append(Cs.col_indices().short())
                val.append(Cs.values())
                nnz += Cs.values().numel()
        self.Q, self.Kt = torch.cat(Qs), torch.cat(Ks)
        self.C = torch.sparse_csr_tensor(torch.cat(crow), torch.cat(col), torch.cat(val), (W * N, W * N))
        self.col_local = torch.cat(col_local)
        print(f'learned graph: {nnz / (W * N):.1f} sources per neuron and window on average, alpha {self.alpha}')

        check = past[:32, -self.P:].to(device)
        with torch.no_grad():
            assert torch.allclose(self.model(check).cpu(), self.predict(None, torch.arange(N))[:32], atol=1e-4)

    def shift(self, keep):
        return shift_operator(self.edge_index[:, keep], self.edge_weight[keep], self.N, 'mean')

    def hop(self, A, y, delta):
        """S_w y for every window w at once. y: [W, N, P]."""
        W, N, P = y.shape
        anatomical = torch.sparse.mm(A, y.transpose(0, 1).reshape(N, W * P)).reshape(N, W, P).transpose(0, 1)
        flat = y.reshape(W * N, P)
        learned = self.C @ flat
        if delta is not None:
            learned = learned + delta @ flat
        return (1 - self.alpha) * anatomical + self.alpha * learned.reshape(W, N, P)

    @torch.no_grad()
    def without_source(self, j):
        """Change of the block-diagonal C when neuron j is masked out of every window's learned graph."""
        N, W = self.N, self.x0.shape[0]
        positions = torch.nonzero(self.col_local == j).flatten()
        rows = torch.searchsorted(self.C.crow_indices(), positions, right=True) - 1
        r_all, c_all, v_all = [], [], []
        for r in rows.split(2048):
            w, i = r // N, r % N
            z = (self.Kt[w] @ self.Q[w, i].unsqueeze(-1)).squeeze(-1) / math.sqrt(self.Q.shape[-1])  # [R, N]
            old = self.learner.normalize_scores(z)
            z[:, j] = -math.inf
            change = self.learner.normalize_scores(z) - old
            k, m = torch.nonzero(change, as_tuple=True)
            r_all.append(r[k])
            c_all.append(w[k] * N + m)
            v_all.append(change[k, m])
        index = torch.stack([torch.cat(r_all), torch.cat(c_all)]) if r_all else torch.zeros(2, 0, dtype=torch.long)
        values = torch.cat(v_all) if v_all else torch.zeros(0)
        return torch.sparse_coo_tensor(index.to(self.device), values.to(self.device), (W * N, W * N)).coalesce().to_sparse_csr()

    @torch.no_grad()
    def predict(self, keep, targets, drop_source=None):
        """Forecasts [W, len(targets)] with only the anatomical edges in `keep`, and without neuron `drop_source` in
        the learned graphs."""
        A = self.A if keep is None else self.shift(keep).to(self.device)
        delta = None if drop_source is None else self.without_source(drop_source)
        y = self.x0
        pred = y @ self.h[0]
        for k in range(1, self.K + 1):
            y = self.hop(A, y, delta)
            pred = pred + y @ self.h[k]
        return pred[:, targets.to(self.device)].cpu()

    def affected(self, keep):
        return torch.arange(self.N)  # the learned graphs connect every neuron within K hops


def per_trial(values, trial_index, n_trials):
    """Sum [W, ...] values within each trial -> [n_trials, ...]."""
    out = torch.zeros(n_trials, *values.shape[1:], dtype=torch.float64)
    return out.index_add_(0, torch.from_numpy(trial_index), values.double())


def bootstrap(trial_sums, counts, n_boot, seed=0):
    """Mean per window and 95% interval from resampling trials. trial_sums: [M, T] -> three [M] arrays."""
    rng = np.random.default_rng(seed)
    T = len(counts)
    w = np.stack([np.bincount(rng.integers(T, size=T), minlength=T) for _ in range(n_boot)])  # [B, T]
    boot = (trial_sums @ w.T) / (counts @ w.T)
    return trial_sums.sum(1) / counts.sum(), np.percentile(boot, 2.5, 1), np.percentile(boot, 97.5, 1)


def run(args):
    if args.learned and args.model != 'adaptive_gvar':
        raise ValueError('--learned only applies to adaptive_gvar')
    if args.model == 'gvar':
        run_dir = MODELS_DIR / 'GVAR' / 'runs' / args.run
        cfg = json.loads((run_dir / 'results.json').read_text())['config']
        data = load_session(cfg['dataset'], cfg['log_transform'], cfg['val_frac'], cfg['test_frac'], use_exog=False)
    elif args.model == 'adaptive_gvar':
        _, saved = adaptive_checkpoint(args.run)
        dataset = saved['dataset']
        if dataset not in DATASETS:
            # adaptive_gvar_sparsemax stores the path of the copy it was trained on (cave_proofread, other neuron order)
            print(f'Checkpoint dataset {dataset} is not in models/GVAR/data.py, using {args.dataset}')
            dataset = args.dataset
        pre = saved['preprocessing']
        data = load_session(dataset, pre['log_transform'], pre['val_frac'], pre['test_frac'], use_exog=False)
    else:
        run_dir = MODELS_DIR / 'stgnn' / 'runs' / args.run
        cfg = json.loads((run_dir / 'results.json').read_text())['config']
        data = load_session(cfg['dataset'], not cfg['no_log_transform'], cfg['val_frac'], cfg['test_frac'], cfg['exog'])

    windows, exog, trial_of = test_windows(data, HISTORY)
    keep_w = np.arange(0, len(windows), args.stride)
    windows, trial_of = windows[keep_w], trial_of[keep_w]
    exog = None if exog is None else exog[keep_w]
    trials, trial_index = np.unique(trial_of, return_inverse=True)
    past, y = windows[:, :HISTORY], windows[:, HISTORY]
    counts = np.bincount(trial_index).astype(np.float64)

    if args.model == 'gvar':
        model = GVARPredictor(run_dir, data, past)
    elif args.model == 'adaptive_gvar':
        model = AdaptiveGVARPredictor(args.run, data, past, torch.device(args.device), args.learned)
    else:
        model = GNNPredictor(run_dir, data, past, exog, torch.device(args.device), args.batch_size)
    N = model.N
    src, dst = model.edge_index
    E = src.numel() if args.max_edges is None else min(args.max_edges, src.numel())
    # with --learned every neuron can send through the learned graphs, not only those with anatomical edges
    senders = list(range(N)) if args.learned else torch.unique(src).tolist()
    senders = senders if args.max_neurons is None else senders[:args.max_neurons]
    print(f'{args.model} {args.run}: {len(windows)} test windows in {len(trials)} trials, '
          f'{src.numel()} edges, {len(senders)} neurons with outgoing edges')

    base_se = (model.predict(None, torch.arange(N)) - y) ** 2  # [W, N]
    base_trial = per_trial(base_se, trial_index, len(trials))  # [T, N]

    start = time.time()
    edge_trial = np.zeros((E, len(trials)))
    for e in range(E):
        keep = torch.ones(src.numel(), dtype=torch.bool)
        keep[e] = False
        i = dst[e:e + 1]
        se = (model.predict(keep, i) - y[:, i]) ** 2
        edge_trial[e] = (per_trial(se, trial_index, len(trials)) - base_trial[:, i]).squeeze(1).numpy()
        if (e + 1) % 200 == 0:
            print(f'  edges {e + 1}/{E}  {time.time() - start:.0f}s')

    node_trial = np.zeros((len(senders), len(trials)))
    for n, j in enumerate(senders):
        keep = src != j
        targets = model.affected(keep)
        pred = model.predict(keep, targets, drop_source=j) if args.learned else model.predict(keep, targets)
        se = (pred - y[:, targets]) ** 2
        node_trial[n] = (per_trial(se, trial_index, len(trials)) - base_trial[:, targets]).sum(1).numpy()
        if (n + 1) % 50 == 0:
            print(f'  neurons {n + 1}/{len(senders)}  {time.time() - start:.0f}s')

    name = args.run if args.run.startswith(args.model) else f'{args.model}_{args.run}'
    out = OUT_DIR / (args.out or name + ('_learned' if args.learned else ''))
    out.mkdir(parents=True, exist_ok=True)
    mean, lo, hi = bootstrap(edge_trial, counts, args.n_boot)
    base_mse = base_se.mean(0).numpy()
    edges = pd.DataFrame({'src': src[:E].numpy(), 'dst': dst[:E].numpy(), 'synapses': model.edge_weight[:E].numpy(),
                          'influence': mean, 'ci_low': lo, 'ci_high': hi, 'target_mse': base_mse[dst[:E].numpy()]})
    edges.to_csv(out / 'edges.csv', index=False)

    mean, lo, hi = bootstrap(node_trial, counts, args.n_boot)
    out_deg = np.bincount(src.numpy(), minlength=N)
    out_syn = np.bincount(src.numpy(), weights=model.edge_weight.numpy(), minlength=N)
    edge_sum = np.bincount(edges['src'], weights=edges['influence'], minlength=N)
    neurons = pd.DataFrame({'neuron': senders, 'out_degree': out_deg[senders], 'out_synapses': out_syn[senders],
                            'sum_edge_influence': edge_sum[senders], 'drop_influence': mean,
                            'drop_ci_low': lo, 'drop_ci_high': hi})
    neurons['redundancy_gap'] = neurons['sum_edge_influence'] - neurons['drop_influence']
    neurons.to_csv(out / 'neurons.csv', index=False)

    np.savez(out / 'trial_sums.npz', edge_trial=edge_trial, node_trial=node_trial, counts=counts,
             senders=np.array(senders), edge_index=model.edge_index.numpy(), edge_weight=model.edge_weight.numpy())
    meta = {'model': args.model, 'run': args.run, 'run_config': model.cfg, 'num_nodes': N, 'stride': args.stride,
            'n_windows': len(windows), 'n_trials': len(trials), 'n_edges_measured': E, 'learned': args.learned,
            'complete': E == src.numel() and args.max_neurons is None,
            'base_test_mse': float(base_mse.mean()), 'seconds': time.time() - start}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'Saved {out}  (base test mse {base_mse.mean():.4f}, {time.time() - start:.0f}s)')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=('gvar', 'adaptive_gvar', 'dcrnn', 'graph_wavenet'), required=True)
    parser.add_argument('--run', required=True,
                        help='run folder in models/GVAR/runs, models/stgnn/runs or (adaptive_gvar) models/GVAR/checkpoints')
    parser.add_argument('--learned', action='store_true',
                        help='adaptive_gvar: also remove each neuron from the learned graphs, for every neuron')
    parser.add_argument('--dataset', default='cave_proofread',
                        help='adaptive_gvar: dataset to use when the checkpoint names a path that does not exist here')
    parser.add_argument('--out', default=None, help='output folder in models/explainability/runs')
    parser.add_argument('--stride', type=int, default=1, help='use every n-th test window (GNNs are slow)')
    parser.add_argument('--max_edges', type=int, default=None, help='only the first n edges, for quick tests')
    parser.add_argument('--max_neurons', type=int, default=None, help='only the first n neurons, for quick tests')
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
