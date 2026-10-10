"""Graph Vector Autoregression, G-VAR with graph filters of order K.

S^k acts on the graph dimension and h_k on the time dimension, 
"""
import torch
import torch.nn as nn


class GVAR(nn.Module):
    def __init__(self, S, P, K):
        super().__init__()
        self.S, self.P, self.K = S, P, K
        self.h = nn.Parameter(torch.zeros(K + 1, P))  # h[k, p-1] = h_{kp}

    def graph_shifts(self, x):
        """x: [M, N] -> [K+1, M, N] with [k] = S^k x for every row of x."""
        out = [x]
        for _ in range(self.K):
            out.append((self.S @ out[-1].T).T)
        return torch.stack(out)

    def features(self, x_past):
        """x_past: [B, P, N], oldest first -> Z: [B, N, K+1, P] with Z[b, n, k, p-1] = (S^k x_{t-p})_n."""
        B, P, N = x_past.shape
        Z = self.graph_shifts(x_past.flip(1).reshape(B * P, N))  # [K+1, B*P, N], lag 1 first
        return Z.reshape(self.K + 1, B, P, N).permute(1, 3, 0, 2)

    def forward(self, x_past):
        """One-step prediction x_t from x_past = [x_{t-P}, ..., x_{t-1}] of shape [B, P, N]. Returns [B, N]."""
        return torch.einsum('bnkp,kp->bn', self.features(x_past), self.h)

    @torch.no_grad()
    def rollout(self, x_past, horizon):
        """Forecast `horizon` steps by feeding predictions back in. Returns [B, horizon, N]."""
        preds = []
        for _ in range(horizon):
            pred = self(x_past)
            preds.append(pred)
            x_past = torch.cat([x_past[:, 1:], pred[:, None]], dim=1)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def fit(self, trials, lam=1e-3):
        """
        Use the closed-form solution to fit the model to a list of trials, each of shape [L, N]. Returns self.
        """
        D = (self.K + 1) * self.P
        A = torch.zeros(D, D, dtype=torch.float64)
        b = torch.zeros(D, dtype=torch.float64)
        n = 0
        for x in trials:
            SkX = self.graph_shifts(x)  # [K+1, L, N]
            L = x.shape[0]
            # F[k, p-1, t] = S^k x_{t-p} for targets t = P..L-1
            F = torch.stack([SkX[:, self.P - p:L - p] for p in range(1, self.P + 1)], dim=1)
            F = F.reshape(D, -1).double()
            y = x[self.P:].reshape(-1).double()
            A += F @ F.T
            b += F @ y
            n += y.numel()
        h = torch.linalg.solve(A / n + lam * torch.eye(D, dtype=torch.float64), b / n)
        self.h.copy_(h.reshape(self.K + 1, self.P).float())
        return self
