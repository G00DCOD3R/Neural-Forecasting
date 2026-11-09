"""Train the activity-dependent Adaptive-GVAR (temporal_adaptive_gvar.py) on a chosen graph.

Same training loop and settings as adaptive_gvar.ipynb, which produced checkpoints/adaptive_gvar_sparsemax
(connectome graph). Use this script for the same model on a degree-preserving random graph, the null for
models/explainability:

    python models/GVAR/train_adaptive.py --graph rewired

Saves checkpoints/<run_name>/best.pt (the epoch with the lowest validation MSE, in the notebook's checkpoint format,
plus the graph it was trained on) and checkpoints/<run_name>/results.json (history and test scores).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

GVAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GVAR_DIR.parent))
sys.path.insert(0, str(GVAR_DIR))
from common import Windows, load_session, split_windows, summarize  # noqa: E402
from data import load_graph, shift_operator, trials  # noqa: E402
from temporal_adaptive_gvar import AdaptiveGVAR, correction_statistics  # noqa: E402

CHECKPOINTS = GVAR_DIR / 'checkpoints'
PREPROCESSING = {'log_transform': True, 'val_frac': 0.15, 'test_frac': 0.15}


def window_dataset(data, which, P):
    """Windows of P + 1 frames inside each trial, as in adaptive_gvar.ipynb."""
    sets = [Windows(x.unfold(0, P + 1, 1).permute(0, 2, 1), None, P) for x in trials(data, which) if len(x) >= P + 1]
    return ConcatDataset(sets)


@torch.no_grad()
def evaluate(model, loader, device):
    """Validation MSE and the sparsity of the learned graph."""
    model.eval()
    squared_error, n_targets, stats, rows = 0.0, 0, {}, 0

    def collect(module, inputs, C):
        nonlocal rows
        n = C.numel() // C.shape[-1]
        rows += n
        for name, value in correction_statistics(C).items():
            stats[name] = stats.get(name, 0.0) + value * n

    hook = model.graph_learner.register_forward_hook(collect)
    try:
        for past, future in loader:
            past, future = past.to(device), future.to(device)
            squared_error += F.mse_loss(model(past).unsqueeze(1), future, reduction='sum').item()
            n_targets += future.numel()
    finally:
        hook.remove()
    return squared_error / n_targets, {name: value / rows for name, value in stats.items()}


@torch.no_grad()
def test_scores(model, data, P, device, batch_size=64):
    windows, _ = split_windows(data, 'test', P, 1)
    preds = [model(windows[i:i + batch_size, :P].to(device)).cpu() for i in range(0, len(windows), batch_size)]
    return {'model': summarize(torch.cat(preds).unsqueeze(1), windows[:, P:]), 'n_windows': len(windows)}


def main(args):
    device = torch.device(args.device)
    run_name = args.run_name or f'adaptive_gvar_{args.activation}_{args.graph}'
    out = CHECKPOINTS / run_name
    out.mkdir(parents=True, exist_ok=True)

    data = load_session(args.dataset, PREPROCESSING['log_transform'], PREPROCESSING['val_frac'],
                        PREPROCESSING['test_frac'], use_exog=False)
    N = data['X'].shape[1]
    edge_index, edge_weight = load_graph(data['data_dir'], N, args.graph, args.graph_seed)
    A = shift_operator(edge_index, edge_weight, N, kind='mean')

    train_set, val_set = window_dataset(data, 'train', args.P), window_dataset(data, 'val', args.P)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {'P': args.P, 'K': args.K, 'alpha': args.alpha, 'hidden_dim': args.hidden_dim,
              'activation': args.activation, 'tau': args.tau}
    model = AdaptiveGVAR(A, **config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'{run_name}: {args.graph} graph ({edge_index.shape[1]} edges), {len(train_set)} train / '
          f'{len(val_set)} val windows, device={device}')

    def save(history):
        torch.save({
            'model_type': 'temporal_adaptive_gvar_v1',
            'state_dict': {name: value.detach().cpu().clone() for name, value in best_state.items()},
            'config': config, 'dataset': args.dataset, 'graph': args.graph, 'graph_seed': args.graph_seed,
            'preprocessing': PREPROCESSING,
            'training': {'lr': args.lr, 'batch_size': args.batch_size, 'seed': args.seed, 'epochs': args.epochs},
            'history': history,
        }, out / 'best.pt')

    history, best_val, best_state = [], float('inf'), None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_error, n_targets = 0.0, 0
        for step, (past, future) in enumerate(train_loader):
            if step == args.max_batches:
                break
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(past).unsqueeze(1), future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            squared_error += loss.item() * future.numel()
            n_targets += future.numel()
        val_mse, stats = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'alpha': args.alpha, 'train_mse': squared_error / n_targets, 'val_mse': val_mse,
                        'activation': args.activation, 'tau': args.tau, **stats})
        print(f'epoch {epoch:3d}  train {squared_error / n_targets:.4f}  val {val_mse:.4f}  '
              f'{time.time() - start:.0f}s', flush=True)
        if val_mse < best_val:
            best_val, best_state = val_mse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save(history)

    model.load_state_dict(best_state)
    save(history)
    results = {'run_name': run_name, 'dataset': args.dataset, 'graph': args.graph, 'graph_seed': args.graph_seed,
               'config': config, 'best_val_mse': best_val, 'history': history,
               'test': test_scores(model, data, args.P, device)}
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print(f"test mse {results['test']['model']['mse'][0]:.4f}  r2 {results['test']['model']['r2'][0]:.4f}")
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='cave_proofread')
    parser.add_argument('--graph', choices=('connectome', 'rewired', 'random'), default='rewired')
    parser.add_argument('--graph_seed', type=int, default=0, help='seed of the random graph, as in GVAR and stgnn')
    parser.add_argument('--run_name', default=None, help='folder in models/GVAR/checkpoints')
    # model and training settings of checkpoints/adaptive_gvar_sparsemax
    parser.add_argument('--P', type=int, default=15)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--activation', choices=('softmax', 'sparsemax'), default='sparsemax')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max_batches', type=int, default=None, help='stop each epoch after n batches, for quick tests')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
