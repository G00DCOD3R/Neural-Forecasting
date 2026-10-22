"""Train DCRNN and Graph WaveNet on a processed MICrONS session.

Both models come from torch-spatiotemporal and are used as graph baselines.
Activity is normalised exactly as in G-VAR. Behaviour is off unless --exog is set,
because G-VAR does not see it.

    python models/stgnn/train.py --smoke
    python models/stgnn/train.py --model dcrnn --dataset cave_proofread
    python models/stgnn/train.py --model graph_wavenet --dataset cave_proofread
    python models/stgnn/train.py --model graph_wavenet --no_learned_adjacency
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

MODELS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS_DIR))
from common import Windows, load_session, naive_baselines, split_windows, summarize  # noqa: E402
from data import load_graph  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / 'runs'


def build_model(name, n_nodes, horizon, exog_size, hidden, ff_size, n_layers, learned_adjacency, dropout):
    from tsl.nn.models import DCRNNModel, GraphWaveNetModel

    if name == 'dcrnn':
        return DCRNNModel(
            input_size=1, output_size=1, horizon=horizon, exog_size=exog_size,
            hidden_size=hidden, kernel_size=2, ff_size=ff_size, n_layers=n_layers,
            cache_support=True, dropout=dropout,
        )
    if name == 'graph_wavenet':
        return GraphWaveNetModel(
            input_size=1, output_size=1, horizon=horizon, exog_size=exog_size,
            hidden_size=hidden, ff_size=ff_size, n_layers=n_layers, n_nodes=n_nodes,
            learned_adjacency=learned_adjacency, dropout=dropout,
        )
    raise ValueError(f'Unknown model: {name}')


def predict(model, past, exog, edge_index, edge_weight):
    """past: [B, T, N] -> [B, horizon, N]."""
    x = past.unsqueeze(-1)
    pred = model(x, edge_index, edge_weight) if exog is None else model(x, edge_index, edge_weight, exog)
    return pred.squeeze(-1)


def batch_tensors(batch, device, use_exog):
    if use_exog:
        past, exog, future = batch
        return past.to(device), exog.to(device), future.to(device)
    past, future = batch
    return past.to(device), None, future.to(device)


def loop(model, loader, edge_index, edge_weight, device, use_exog, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    preds, targets = [], []
    for batch in loader:
        past, exog, future = batch_tensors(batch, device, use_exog)
        if training and past.shape[0] < 2:
            continue  # batch norm in Graph WaveNet is undefined for a single window
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            pred = predict(model, past, exog, edge_index, edge_weight)
            loss = F.mse_loss(pred, future)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * future.shape[0]
        count += future.shape[0]
        if not training:
            preds.append(pred.detach().cpu())
            targets.append(future.detach().cpu())
    mean = total / max(count, 1)
    if training:
        return mean, None, None
    return mean, torch.cat(preds), torch.cat(targets)


def smoke(device):
    """One forward and backward pass of each architecture on a tiny random graph."""
    n_nodes, history, horizon, n_edges = 12, 15, 1, 24
    past = torch.randn(4, history, n_nodes, device=device)
    future = torch.randn(4, horizon, n_nodes, device=device)
    exog = torch.randn(4, history, 5, device=device)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)
    edge_weight = torch.rand(n_edges, device=device) + 0.1
    checks = [
        ('dcrnn', 0, False, None),
        ('dcrnn', 5, False, exog),
        ('graph_wavenet', 0, False, None),
        ('graph_wavenet', 5, True, exog),
    ]
    for name, exog_size, learned, u in checks:
        layers = 8 if name == 'graph_wavenet' else 2
        model = build_model(name, n_nodes, horizon, exog_size, 8, 16, layers, learned, 0.0).to(device)
        pred = predict(model, past, u, edge_index, edge_weight)
        assert pred.shape == future.shape, (name, tuple(pred.shape))
        F.mse_loss(pred, future).backward()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{name:14s} exog={exog_size} learned_adj={learned!s:5s} params={n_params:6d} ok')

    series = torch.randn(8, history + horizon, n_nodes)
    behaviour = torch.randn(8, history + horizon, 5)
    model = build_model('graph_wavenet', n_nodes, horizon, 5, 8, 16, 8, True, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(Windows(series, behaviour, history), batch_size=4, shuffle=True)
    plain = build_model('dcrnn', n_nodes, horizon, 0, 8, 16, 1, False, 0.0).to(device)
    plain_loader = DataLoader(Windows(series, None, history), batch_size=4)
    _, pred, _ = loop(plain, plain_loader, edge_index, edge_weight, device, False)
    assert pred.shape == (8, horizon, n_nodes)

    train_mse, _, _ = loop(model, train_loader, edge_index, edge_weight, device, True, optimizer)
    _, pred, target = loop(model, DataLoader(Windows(series, behaviour, history), batch_size=4),
                           edge_index, edge_weight, device, True)
    assert pred.shape == (8, horizon, n_nodes)
    stats = summarize(pred, target)
    assert set(stats) == {'mse', 'rmse', 'mae', 'pearson', 'r2'}
    print(f'train loop mse {train_mse:.4f}  eval r {stats["pearson"][0]:.3f}')
    print('Smoke test passed.')


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data = load_session(args.dataset, not args.no_log_transform, args.val_frac, args.test_frac, args.exog)
    n_nodes = data['X'].shape[1]
    edge_index, edge_weight = load_graph(data['data_dir'], n_nodes, args.graph, args.seed)
    edge_index, edge_weight = edge_index.to(device), edge_weight.float().to(device)

    train_x, train_u = split_windows(data, 'train', args.history, args.horizon)
    val_x, val_u = split_windows(data, 'val', args.history, args.horizon)
    test_x, test_u = split_windows(data, 'test', args.history, args.horizon)
    exog_size = 0 if train_u is None else train_u.shape[-1]
    learned = args.model == 'graph_wavenet' and not args.no_learned_adjacency
    model = build_model(
        args.model, n_nodes, args.horizon, exog_size, args.hidden, args.ff_size,
        args.layers, learned, args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{args.model} params={n_params} windows train/val/test '
          f'{len(train_x)}/{len(val_x)}/{len(test_x)} device={device}')

    def loader(x, u, shuffle):
        batch = min(args.batch_size, len(x))
        return DataLoader(Windows(x, u, args.history), batch_size=batch, shuffle=shuffle,
                          drop_last=shuffle and len(x) > batch)

    train_loader = loader(train_x, train_u, True)
    val_loader = loader(val_x, val_u, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, best_state, patience = float('inf'), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_mse, _, _ = loop(model, train_loader, edge_index, edge_weight, device, args.exog, optimizer)
        val_mse, _, _ = loop(model, val_loader, edge_index, edge_weight, device, args.exog)
        history.append({'epoch': epoch, 'train_mse': train_mse, 'val_mse': val_mse})
        print(f'epoch {epoch:3d}  train {train_mse:.4f}  val {val_mse:.4f}')
        if val_mse < best_val:
            best_val, patience = val_mse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(best_state)
    _, pred, target = loop(model, loader(test_x, test_u, False), edge_index, edge_weight, device, args.exog)
    test = {'model': summarize(pred, target), **naive_baselines(test_x, args.history), 'n_windows': len(test_x)}
    for name in ('model', 'copy_last', 'mean'):
        m = test[name]
        print(f"test {name:10s} h=1 mse {m['mse'][0]:.4f} mae {m['mae'][0]:.4f} "
              f"r {m['pearson'][0]:.4f} r2 {m['r2'][0]:.4f}")

    run_dir = RUNS_DIR / (args.run_name or f'{args.dataset}_{args.model}_{args.graph}')
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'config': vars(args), 'n_params': n_params, 'learned_adjacency': learned,
        'best_val_mse': best_val, 'history': history, 'test': test,
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(payload, f, indent=2)
    torch.save(best_state, run_dir / 'best.pt')
    print(f'Saved {run_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=('dcrnn', 'graph_wavenet'), default='dcrnn')
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--graph', choices=('connectome', 'random', 'rewired'), default='connectome')
    parser.add_argument('--history', type=int, default=15)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--ff_size', type=int, default=256)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--test_frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--exog', action='store_true', help='include treadmill and pupil')
    parser.add_argument('--no_log_transform', action='store_true')
    parser.add_argument('--no_learned_adjacency', action='store_true',
                        help='Graph WaveNet uses only the fixed connectome')
    parser.add_argument('--smoke', action='store_true', help='check both models on synthetic data and exit')
    args = parser.parse_args()
    if args.layers is None:
        args.layers = 1 if args.model == 'dcrnn' else 8
    if args.dropout is None:
        args.dropout = 0.0 if args.model == 'dcrnn' else 0.3
    return args


if __name__ == '__main__':
    args = parse_args()
    try:
        smoke(torch.device(args.device)) if args.smoke else run(args)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Missing dependency. Install with:\n'
            '  python -m pip install -r models/stgnn/requirements.txt\n'
            f'({exc})'
        )
