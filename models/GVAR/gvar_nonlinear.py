# Nonlinear G-VAR: a fitted (frozen) G-VAR plus stacked graph filters with nonlinearities between them.
#   H^0 = [x_{t-1}, ..., x_{t-P}] per node, plus W_u [u_t, ..., u_{t-Q+1}] (shared by all nodes) with exogenous inputs
#   H^l = σ(sum_k S^k H^{l-1} W_k^l + b^l), or tanh(A) * sigmoid(B) with two such filters when gated
#   x_t = G-VAR(x) + sum_k S^k H^L w_k
# Trained by Adam on the one-step loss; the readout starts at zero, so epoch 0 is the G-VAR.
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from gvar import GVAR


# sum_k S^k H W_k + b: [B, N, F_in] -> [B, N, F_out]
class GraphFilter(nn.Module):
    def __init__(self, K, f_in, f_out):
        super().__init__()
        self.W = nn.Parameter(torch.randn(K + 1, f_in, f_out) / math.sqrt(f_in * (K + 1)))
        self.b = nn.Parameter(torch.zeros(f_out))

    def forward(self, S, H):
        B, N, F = H.shape
        out, Z = H @ self.W[0], H
        for k in range(1, len(self.W)):
            Z = (S @ Z.permute(1, 0, 2).reshape(N, B * F)).reshape(N, B, F).permute(1, 0, 2)
            out = out + Z @ self.W[k]
        return out + self.b


class NonlinearGVAR(nn.Module):
    def __init__(self, linear, hidden=32, layers=2, K=None, gated=False, exo_dim=8):
        super().__init__()
        self.linear = linear.requires_grad_(False)
        K = linear.K if K is None else K
        self.gated = gated
        self.exo_proj = nn.Linear(linear.Q * linear.n_exo, exo_dim) if linear.n_exo and exo_dim else None
        dims = [linear.P + (exo_dim if self.exo_proj else 0)] + [hidden] * layers
        self.filters = nn.ModuleList(GraphFilter(K, a, 2 * b if gated else b) for a, b in zip(dims[:-1], dims[1:]))
        self.readout = GraphFilter(K, dims[-1], 1)
        nn.init.zeros_(self.readout.W)

    # what evaluation and rollout need, delegated to the G-VAR
    P = property(lambda self: self.linear.P)
    Q = property(lambda self: self.linear.Q)
    n_exo = property(lambda self: self.linear.n_exo)
    floor = property(lambda self: self.linear.floor)
    rollout = GVAR.rollout

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # move the G-VAR's non-parameter tensors too
    def to(self, device):
        super().to(device)
        self.linear.S = self.linear.S.to(device)
        if self.linear.floor is not None:
            self.linear.floor = self.linear.floor.to(device)
        return self

    def forward(self, x_past, u_past=None):
        S = self.linear.S
        H = x_past.flip(1).permute(0, 2, 1)  # [B, N, P], lag 1 first
        if self.exo_proj is not None:
            u = self.exo_proj(u_past.flip(1).reshape(len(u_past), -1))  # [B, exo_dim]
            H = torch.cat([H, u[:, None].expand(-1, H.shape[1], -1)], -1)
        for f in self.filters:
            H = f(S, H)
            if self.gated:
                a, b = H.chunk(2, -1)
                H = torch.tanh(a) * torch.sigmoid(b)
            else:
                H = torch.relu(H)
        return self.linear(x_past, u_past) + self.readout(S, H).squeeze(-1)


def device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


# Adam on the one-step MSE, early-stopped on val_score(model) (lower is better).
# Returns the best model (epoch 0 = the G-VAR), the history and the best epoch.
def train(model, data, val_score, epochs=30, lr=1e-3, weight_decay=1e-4, batch_size=64, patience=3, seed=0,
          verbose=True):
    torch.manual_seed(seed)
    dev = next(p.device for p in model.parameters() if p.requires_grad)
    X, U = data['X'].to(dev), data['U'].to(dev)
    P, Q = model.P, model.Q
    t0 = max(P, Q - 1) if model.n_exo else P
    # training targets with t0 past frames in their trial (rows are in recording order)
    tid = data['trial_id']
    starts = np.r_[0, np.flatnonzero(np.diff(tid)) + 1]
    pos = np.arange(len(tid)) - starts[np.searchsorted(starts, np.arange(len(tid)), side='right') - 1]
    targets = torch.from_numpy(np.flatnonzero((data['split'][tid] == 'train') & (pos >= t0))).to(dev)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    history = [{'epoch': 0, 'train_mse': None, **val_score(model)}]
    best = (history[0]['val_mse'], copy.deepcopy(model.state_dict()), 0)
    if verbose:
        print(f"  epoch  0  val {history[0]['val_mse']:.4f} (h=1 {history[0]['val_mse_h1']:.4f})  G-VAR")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for idx in targets[torch.randperm(len(targets), device=dev)].split(batch_size):
            x_past = X[idx[:, None] + torch.arange(-P, 0, device=dev)]
            u_past = U[idx[:, None] + torch.arange(-Q + 1, 1, device=dev)] if model.n_exo else None
            loss = ((model(x_past, u_past) - X[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        history.append({'epoch': epoch, 'train_mse': total / len(targets), **val_score(model)})
        if verbose:
            h = history[-1]
            print(f"  epoch {epoch:2d}  train {h['train_mse']:.4f}  val {h['val_mse']:.4f} (h=1 {h['val_mse_h1']:.4f})")
        if history[-1]['val_mse'] < best[0]:
            best = (history[-1]['val_mse'], copy.deepcopy(model.state_dict()), epoch)
        elif epoch - best[2] >= patience:
            break
    model.load_state_dict(best[1])
    return model, history, best[2]
