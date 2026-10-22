"""Data loading for G-VAR: chronological split, normalisation, graph shift operator."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    'cave_proofread': REPO_ROOT / 'processed' / 'session_4_7_cave_proofread',
    'cave_all': REPO_ROOT / 'processed' / 'session_4_7_cave_all',
}


def make_split(trial_start, val_frac=0.15, test_frac=0.15):
    """Split trials by time: train on the start of the session, validate on the next part, test on the end.

    The cuts fall between trials, at the given fractions of the session duration.
    """
    t = (trial_start - trial_start[0]) / (trial_start[-1] - trial_start[0])
    return np.where(t >= 1 - test_frac, 'test', np.where(t >= 1 - test_frac - val_frac, 'val', 'train'))


def load_data(dataset, log_transform=True, val_frac=0.15, test_frac=0.15):
    """Load session, normalise the responses, and split trials chronologically into train/val/test."""
    data_dir = DATASETS.get(dataset, dataset)
    X = np.load(os.path.join(data_dir, 'X_features.npy')).astype(np.float32)
    rows = pd.read_csv(os.path.join(data_dir, 'conditions.csv'))
    assert 'trial_id' in rows, f"{data_dir} was built before trial ids and timestamps were stored, rebuild it"
    assert rows['time'].is_monotonic_increasing, "rows of X are not in recording order"

    trial_id = rows['trial_id'].values
    trial_start = rows.groupby('trial_id')['time'].first().values
    split = make_split(trial_start, val_frac, test_frac)
    train_rows = split[trial_id] == 'train'

    # log1p responses to account for a  few bursts from dominating the MSE
    if log_transform:
        X = np.log1p(np.clip(X, 0, None))
    mu, sd = X[train_rows].mean(0), X[train_rows].std(0) + 1e-6
    X = (X - mu) / sd

    return {'X': torch.from_numpy(X), 'trial_id': trial_id, 'split': split, 'trial_start': trial_start,
            'data_dir': str(data_dir)}


def trials(data, which):
    """List of [L, N] tensors, one per trial of the given split."""
    ids = np.flatnonzero(data['split'] == which)
    return [data['X'][data['trial_id'] == t] for t in ids]


def load_graph(data_dir, num_nodes, graph='connectome', seed=0):
    """Return (edge_index, edge_weight) with edge_index[0] = presynaptic, edge_index[1] = postsynaptic.

    connectome: the synaptic edges, weighted by synapse count
    random:     as many edges as the connectome between random pairs, shuffled weights
    rewired:    the connectome with targets shuffled, so every neuron keeps its in- and out-degree
    """
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'))
    edge_weight = torch.load(os.path.join(data_dir, 'edge_weight.pt'))
    if graph == 'connectome':
        return edge_index, edge_weight
    if graph == 'random':
        g = torch.Generator().manual_seed(seed)
        E = edge_index.shape[1]
        flat = torch.randperm(num_nodes * (num_nodes - 1), generator=g)[:E]
        src, off = flat // (num_nodes - 1), flat % (num_nodes - 1)
        dst = off + (off >= src).long()  # skip the diagonal
        return torch.stack([src, dst]), edge_weight[torch.randperm(E, generator=g)]
    if graph == 'rewired':
        return rewire(edge_index, num_nodes, seed), edge_weight
    raise ValueError(f"Unknown graph: {graph}")


def rewire(edge_index, num_nodes, seed=0, max_rounds=1000):
    """Degree-preserving randomisation: shuffle the edge targets, then swap away self-loops and repeated edges.

    Sources stay in place, so out-degree and each edge's weight are kept; targets are permuted, so in-degree is kept.
    """
    rng = np.random.default_rng(seed)
    src, dst = edge_index.numpy().copy()
    dst = rng.permutation(dst)
    for _ in range(max_rounds):
        key = src.astype(np.int64) * num_nodes + dst
        _, first = np.unique(key, return_index=True)
        bad = np.ones(len(key), bool)
        bad[first] = False  # later copies of an edge are bad
        bad |= src == dst
        if not bad.any():
            return torch.from_numpy(np.stack([src, dst])).long()
        for i in np.flatnonzero(bad):
            j = rng.integers(len(dst))
            dst[i], dst[j] = dst[j], dst[i]
    raise RuntimeError('Could not rewire without self-loops or repeated edges')


def shift_operator(edge_index, edge_weight, num_nodes, kind='mean', undirected=False):
    """Sparse graph shift operator S 

    mean: rows normalised by weighted in-degree.
    adj:  raw synapse counts divided by the largest absolute row sum.
    """
    src, dst = edge_index
    w = edge_weight.float()
    if undirected:
        src, dst, w = torch.cat([src, dst]), torch.cat([dst, src]), torch.cat([w, w])
    row_sum = torch.zeros(num_nodes).index_add_(0, dst, w)
    if kind == 'mean':
        w = w / row_sum[dst]
    elif kind == 'adj':
        w = w / row_sum.max()
    else:
        raise ValueError(f"Unknown shift operator: {kind}")
    return torch.sparse_coo_tensor(torch.stack([dst, src]), w, (num_nodes, num_nodes), check_invariants=True).coalesce()
