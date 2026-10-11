"""GVAR with a static correction scored from training-only pair correlations."""
import copy
import math

import torch
from torch import nn
from tqdm import tqdm

from gvar import GVAR
from proximity_gvar import evaluate_trials


def candidate_edges(positions_um, anatomical_edges, neighbours=5, radius_um=150):
    """Return [source, target]: nearest eligible sources per target, no self/known edges."""
    positions = torch.as_tensor(positions_um, dtype=torch.float64)
    N = len(positions)
    if N < 2 or positions.shape != (N, 3) or not torch.isfinite(positions).all():
        raise ValueError('Expected finite positions of shape [N,3], N >= 2')
    if neighbours < 1 or radius_um <= 0:
        raise ValueError('neighbours and radius_um must be positive')
    distance = torch.cdist(positions, positions, compute_mode='donot_use_mm_for_euclid_dist')
    distance.fill_diagonal_(torch.inf)
    src, dst = anatomical_edges
    distance[dst, src] = torch.inf  # Exclude this direction only.
    distances, sources = distance.topk(min(neighbours, N - 1), largest=False, dim=1)
    targets = torch.arange(N)[:, None].expand_as(sources)
    eligible = distances < radius_um
    return torch.stack([sources[eligible], targets[eligible]])


@torch.no_grad()
def correlation_features(calibration_trials, edges):
    """Columns: r(source,target), r(source[t-1],target[t]), r(target[t-1],source[t]).

    Pass training trials only. Pool within-trial pairs; never join trial boundaries.
    Undefined correlations (e.g. constant signals) become zero.
    """
    if not calibration_trials or any(len(x) < 2 for x in calibration_trials):
        raise ValueError('Calibration requires nonempty trials of at least two frames')
    src, dst = edges
    moments = torch.zeros(6, 3, len(src), dtype=torch.float64)
    for x in calibration_trials:
        a, b = x[:, src].double(), x[:, dst].double()
        for k, (u, v) in enumerate([(a, b), (a[:-1], b[1:]), (b[:-1], a[1:])]):
            valid = torch.isfinite(u) & torch.isfinite(v)
            u, v = u.masked_fill(~valid, 0), v.masked_fill(~valid, 0)
            moments[:, k] += torch.stack([valid.sum(0), u.sum(0), v.sum(0),
                                          u.square().sum(0), v.square().sum(0), (u * v).sum(0)])
    n, su, sv, suu, svv, suv = moments
    vu, vv = (suu - su.square() / n).clamp_min(0), (svv - sv.square() / n).clamp_min(0)
    corr = (suv - su * sv / n) / (vu * vv).sqrt()
    corr[n < 3] = 0
    return torch.nan_to_num(corr, nan=0, posinf=0, neginf=0).clamp(-1, 1).T.float()


class AdaptiveGVAR(GVAR):
    """Same GVAR forecasting equation; only its shared sparse shift is learned."""

    def __init__(self, A, candidates, features, P=15, K=3, alpha=.1, learn_edges=True):
        if not 0 <= alpha < 1 or P < 1 or K < 1:
            raise ValueError('Require 0 <= alpha < 1, P >= 1, K >= 1')
        if candidates.shape[0] != 2 or candidates.shape[1] == 0 or features.shape != (candidates.shape[1], 3):
            raise ValueError('Expected nonempty candidates [2,E] and features [E,3]')
        super().__init__(A, P, K)
        nn.init.normal_(self.h, mean=0., std=.01)  # Nonzero graph terms from the start.
        self.register_buffer('A', A.detach().coalesce())  # Already row-normalized.
        self.register_buffer('candidates', candidates.clone().long())
        self.register_buffer('calibration_features', features.detach().clone().float())
        self.scorer = (nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
                       if learn_edges else None)
        self.alpha = alpha
        self.update_graph()

    def update_graph(self):
        """Rebuild from fixed calibration features, never from a forecasting window."""
        if self.alpha == 0:
            self.S = self.A
            return self.S
        src, dst = self.candidates
        weights = (self.scorer(self.calibration_features).flatten() if self.scorer is not None
                   else self.calibration_features.new_ones(len(src)))
        row_sum = weights.new_zeros(self.A.shape[0]).index_add(0, dst, weights)
        normalized = weights / row_sum[dst].clamp_min(1e-12)
        correction = torch.sparse_coo_tensor(torch.stack([dst, src]), normalized, self.A.shape,
                                             check_invariants=True).coalesce()
        self.S = ((1 - self.alpha) * self.A + self.alpha * correction).coalesce()
        return self.S

    def fit(self, train_trials, val_trials, epochs=5, lr=1e-3, lam=1e-3,
            batch_size=64, progress=True):
        """Joint Adam updates from initialization; fixed features, one graph per batch."""
        if epochs < 1 or lr <= 0 or lam < 0 or batch_size < 1 or not train_trials or not val_trials:
            raise ValueError('Require positive epochs/lr/batch_size, nonnegative lam and nonempty splits')
        if any(len(x) <= self.P for x in train_trials):
            raise ValueError('Every training trial must be longer than P')
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_batches = sum(math.ceil((len(x) - self.P) / batch_size) for x in train_trials)

        def val_mse():
            return evaluate_trials(self, val_trials, context=max(15, self.P), horizon=1,
                                   progress=False)['gvar']['mse'][0]

        self.eval()
        with torch.no_grad():
            self.update_graph()
        self.history = [{'epoch': 0, 'train_mse': None, 'val_mse': val_mse(),
                         'mlp_grad_norm': None, 'graph_h_grad_norm': None}]
        best_loss, best = self.history[0]['val_mse'], copy.deepcopy(self.state_dict())
        bar = tqdm(range(1, epochs + 1), desc='Joint GVAR', unit='epoch', disable=not progress)
        for epoch in bar:
            self.train()
            squared_error, n_targets = 0., 0
            mlp_grad_norm, graph_h_grad_norm, updates = 0., 0., 0
            with tqdm(total=n_batches, desc=f'Epoch {epoch}', unit='batch', leave=False,
                      disable=not progress) as batches:
                for index in torch.randperm(len(train_trials)).tolist():
                    x = train_trials[index]
                    windows = x.unfold(0, self.P + 1, 1).permute(0, 2, 1)
                    for indices in torch.randperm(len(windows)).split(batch_size):
                        batch = windows[indices]
                        optimizer.zero_grad()
                        self.update_graph()  # Same calibration features for every update/window.
                        error = (self(batch[:, :-1]) - batch[:, -1]).square()
                        loss = error.mean() + lam * self.h.square().sum()
                        loss.backward()
                        graph_h_grad_norm += self.h.grad[1:].norm().item()
                        if self.scorer is not None:
                            mlp_grad_norm += math.sqrt(sum(p.grad.square().sum().item() for p in
                                                          self.scorer.parameters() if p.grad is not None))
                        optimizer.step()  # Jointly updates h and the edge scorer.
                        squared_error += error.detach().sum().item()
                        n_targets += error.numel()
                        updates += 1
                        batches.update()
                        batches.set_postfix(mse=f'{squared_error / n_targets:.5f}')
            self.eval()
            with torch.no_grad():
                self.update_graph()  # Static graph for all validation predictions.
            score = val_mse()
            self.history.append({'epoch': epoch, 'train_mse': squared_error / n_targets,
                                 'val_mse': score, 'mlp_grad_norm': mlp_grad_norm / updates,
                                 'graph_h_grad_norm': graph_h_grad_norm / updates})
            bar.set_postfix(train_mse=f'{squared_error / n_targets:.5f}', val_mse=f'{score:.5f}')
            if score < best_loss:
                best_loss, best = score, copy.deepcopy(self.state_dict())
        self.load_state_dict(best)
        self.eval()
        with torch.no_grad():
            self.update_graph()
        return self
