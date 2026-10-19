"""Copy-last, training-mean, and shared GRU on the same task as G-VAR.

The activity split, log1p transform, training-only normalisation, and 15-frame
context match G-VAR. Behaviour is off unless --exog is set. The GRU predicts
one step; compare it with `models/GVAR/train.py --horizon 1`.

    python models/baselines.py
    python models/baselines.py --exog
    python models/baselines.py --dataset cave_all
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_DIR))
from common import Windows, load_session, naive_baselines, split_windows, summarize  # noqa: E402

HISTORY = 15  # G-VAR evaluates every model with this context length
HORIZON = 1


class SharedGRU(nn.Module):
    """One GRU applied independently to every neuron, with weights shared across neurons."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def predict_gru(model, past, exog):
    """past: [B, T, N] -> [B, 1, N]. Each neuron sees only its own history."""
    batch, time, nodes = past.shape
    x = past.permute(0, 2, 1).reshape(batch * nodes, time, 1)
    if exog is not None:
        features = exog.shape[-1]
        u = exog.unsqueeze(1).expand(batch, nodes, time, features).reshape(batch * nodes, time, features)
        x = torch.cat([x, u], dim=-1)
    return model(x).view(batch, 1, nodes)


def batch_tensors(batch, device, use_exog):
    if use_exog:
        past, exog, future = batch
        return past.to(device), exog.to(device), future.to(device)
    past, future = batch
    return past.to(device), None, future.to(device)


def loop(model, loader, device, use_exog, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    preds, targets = [], []
    for batch in loader:
        past, exog, future = batch_tensors(batch, device, use_exog)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            pred = predict_gru(model, past, exog)
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


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    data = load_session(args.dataset, not args.no_log_transform, args.val_frac, args.test_frac, args.exog)
    train_x, train_u = split_windows(data, 'train', HISTORY, HORIZON)
    val_x, val_u = split_windows(data, 'val', HISTORY, HORIZON)
    test_x, test_u = split_windows(data, 'test', HISTORY, HORIZON)
    input_dim = 1 if train_u is None else 1 + train_u.shape[-1]
    model = SharedGRU(input_dim, args.hidden).to(device)
    print(f'GRU params={sum(p.numel() for p in model.parameters())} '
          f'windows train/val/test {len(train_x)}/{len(val_x)}/{len(test_x)} device={device}')

    def loader(x, u, shuffle):
        return DataLoader(Windows(x, u, HISTORY), batch_size=min(args.batch_size, len(x)), shuffle=shuffle)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, best_state, patience = float('inf'), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_mse, _, _ = loop(model, loader(train_x, train_u, True), device, args.exog, optimizer)
        val_mse, _, _ = loop(model, loader(val_x, val_u, False), device, args.exog)
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
    _, pred, target = loop(model, loader(test_x, test_u, False), device, args.exog)
    test = {'gru': summarize(pred, target), **naive_baselines(test_x, HISTORY), 'n_windows': len(test_x)}
    for name in ('gru', 'copy_last', 'mean'):
        m = test[name]
        print(f"test {name:10s} h=1 mse {m['mse'][0]:.4f} mae {m['mae'][0]:.4f} "
              f"r {m['pearson'][0]:.4f} r2 {m['r2'][0]:.4f}")

    run_dir = Path('results') / (args.run_name or f"baselines_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {'config': vars(args), 'history_len': HISTORY, 'horizon': HORIZON,
               'best_val_mse': best_val, 'history': history, 'test': test}
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(payload, f, indent=2)
    torch.save(best_state, run_dir / 'best_gru.pt')
    print(f'Saved {run_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--test_frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--exog', action='store_true', help='include treadmill and pupil')
    parser.add_argument('--no_log_transform', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
