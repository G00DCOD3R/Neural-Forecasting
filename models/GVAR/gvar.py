# G-VAR with graph filters of order K, optionally leaky and with exogenous inputs:
#   x_t = sum_{k,p} h_kp S^k x_{t-p}
#       + sum_{k,p} g_kp S^k relu(x_{t-p})     leaky: slope h below 0, h + g above
#       + B [u_t, ..., u_{t-Q+1}] + c           per-neuron exogenous readout and intercept
# Linear in its parameters, so fitted in closed form.
import numpy as np
import torch
import torch.nn as nn


class GVAR(nn.Module):
    def __init__(self, S, P, K, leaky=False, n_exo=0, Q=1, intercept=False):
        super().__init__()
        self.S, self.P, self.K, self.n_exo, self.Q = S, P, K, n_exo, Q
        N = S.shape[0]
        self.h = nn.Parameter(torch.zeros(K + 1, P))  # h[k, p-1] = h_{kp}
        self.g = nn.Parameter(torch.zeros(K + 1, P)) if leaky else None
        self.B = nn.Parameter(torch.zeros(N, Q * n_exo)) if n_exo else None  # columns: lag 0 first, E per lag
        self.c = nn.Parameter(torch.zeros(N)) if n_exo or intercept else None
        self.floor = None  # [N], if set rollout predictions are clamped at it

    # [C, K+1, P], one row of taps per input channel
    @property
    def taps(self):
        return torch.stack([self.h, self.g]) if self.g is not None else self.h[None]

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    # [M, N] -> [C, M, N]: x, and relu(x) when leaky
    def channels(self, x):
        return torch.stack([x, torch.relu(x)]) if self.g is not None else x[None]

    # [M, N] -> [K+1, M, N], [k] = S^k x
    def graph_shifts(self, x):
        out = [x]
        for _ in range(self.K):
            out.append((self.S @ out[-1].T).T)
        return torch.stack(out)

    # x_past [B, P, N], oldest first -> [B, N, C, K+1, P], [b, n, c, k, p-1] = (S^k σ_c(x_{t-p}))_n
    def features(self, x_past):
        B, P, N = x_past.shape
        x = self.channels(x_past.flip(1).reshape(B * P, N))  # [C, B*P, N], lag 1 first
        Z = torch.stack([self.graph_shifts(xc) for xc in x])  # [C, K+1, B*P, N]
        return Z.reshape(len(Z), self.K + 1, B, P, N).permute(2, 4, 0, 1, 3)

    # One step: x_past [B, P, N], u_past [B, Q, E] = u_{t-Q+1..t} -> x_t [B, N]
    def forward(self, x_past, u_past=None):
        out = torch.einsum('bnckp,ckp->bn', self.features(x_past), self.taps)
        if self.n_exo:
            out = out + u_past.flip(1).reshape(len(u_past), -1) @ self.B.T
        if self.c is not None:
            out = out + self.c
        return out

    # Recursive forecast [B, horizon, N]; u [B, Q-1+horizon, E] = u_{t-Q+1..t+horizon-1}
    @torch.no_grad()
    def rollout(self, x_past, horizon, u=None):
        preds = []
        for j in range(horizon):
            pred = self(x_past, u[:, j:j + self.Q] if self.n_exo else None)
            if self.floor is not None:
                pred = torch.maximum(pred, self.floor)
            preds.append(pred)
            x_past = torch.cat([x_past[:, 1:], pred[:, None]], dim=1)
        return torch.stack(preds, dim=1)

    # Least-squares statistics over trials [L, N] (and [L, E]), targets t0..L-1 of each trial.
    # lead > 0 (direct forecasting): target x_t from x_{t-lead-1}, ...; inputs not known ahead are then held at the
    # origin t-lead, as in evaluation.
    @torch.no_grad()
    def statistics(self, trials, exo_trials=None, lead=0, known_ahead=None):
        t0 = max(self.P + lead, self.Q - 1) if self.n_exo else self.P + lead
        D = self.taps.numel()
        st = {'FF': torch.zeros(D, D, dtype=torch.float64), 'Fy': torch.zeros(D, dtype=torch.float64), 'T': 0}
        for i, x in enumerate(trials):
            L, N = x.shape
            if L <= t0:
                continue
            SkX = torch.stack([self.graph_shifts(xc) for xc in self.channels(x)])  # [C, K+1, L, N]
            # F[c, k, p-1, t] = S^k σ_c(x_{t-p})
            F = torch.stack([SkX[:, :, t0 - lead - p:L - lead - p] for p in range(1, self.P + 1)], dim=2)
            F = F.reshape(D, L - t0, N)
            y = x[t0:]
            F2 = F.reshape(D, -1).double()
            st['FF'] += F2 @ F2.T
            st['Fy'] += F2 @ y.reshape(-1).double()
            st['T'] += L - t0
            if self.c is not None:
                # per-neuron design rows [u_t, ..., u_{t-Q+1}, 1]
                lags = [self.held(exo_trials[i], t0, q, lead, known_ahead) for q in range(self.Q)] if self.n_exo else []
                U = torch.cat(lags + [torch.ones(L - t0, 1)], 1)
                E1 = U.shape[1]
                if 'UU' not in st:
                    st.update(UU=torch.zeros(E1, E1, dtype=torch.float64), Uy=torch.zeros(E1, N, dtype=torch.float64),
                              UF=torch.zeros(E1, D, N, dtype=torch.float64))
                st['UU'] += (U.T @ U).double()
                st['Uy'] += (U.T @ y).double()
                st['UF'] += (U.T @ F.permute(1, 0, 2).reshape(L - t0, D * N)).reshape(E1, D, N).double()
        return st

    # Rows t0..L-1 of lag q of u, with columns not known ahead taken at the origin t - lead if that is earlier
    @staticmethod
    def held(u, t0, q, lead, known_ahead):
        lag = u[t0 - q:len(u) - q]
        if q >= lead or known_ahead is None or all(known_ahead):
            return lag
        lag = lag.clone()
        cols = torch.from_numpy(~np.asarray(known_ahead, bool))
        lag[:, cols] = u[t0 - lead:len(u) - lead][:, cols]
        return lag

    # Ridge solve: (1/TN) |err|^2 + lam |taps|^2 + (lam_exo/N) sum_n |b_n|^2, intercept unpenalised.
    # b_n is eliminated per neuron (shared G = U'U/T + lam_exo I), leaving a Schur complement system for the taps.
    @torch.no_grad()
    def solve(self, st, lam=1e-3, lam_exo=1.0):
        N = self.S.shape[0]
        T, D = st['T'], st['FF'].shape[0]
        A = st['FF'] / T + N * lam * torch.eye(D, dtype=torch.float64)
        b = st['Fy'] / T
        if self.c is not None:
            E1 = st['UU'].shape[0]
            G = st['UU'] / T + lam_exo * torch.diag(torch.cat([torch.ones(E1 - 1), torch.zeros(1)])).double()
            C, R = st['UF'] / T, st['Uy'] / T  # [E1, D, N], [E1, N]
            GiC = torch.linalg.solve(G, C.reshape(E1, D * N)).reshape(E1, D, N)
            GiR = torch.linalg.solve(G, R)
            A = A - torch.einsum('edn,efn->df', C, GiC)
            b = b - torch.einsum('edn,en->d', C, GiR)
        taps = torch.linalg.solve(A, b)
        self.h.copy_(taps[:self.h.numel()].reshape(self.K + 1, self.P).float())
        if self.g is not None:
            self.g.copy_(taps[self.h.numel():].reshape(self.K + 1, self.P).float())
        if self.c is not None:
            Bc = GiR - torch.einsum('edn,d->en', GiC, taps)  # [E1, N]
            if self.n_exo:
                self.B.copy_(Bc[:-1].T.float())
            self.c.copy_(Bc[-1].float())
        return self

    # Closed-form fit to trials of shape [L, N]
    def fit(self, trials, lam=1e-3, lam_exo=1.0, exo_trials=None):
        return self.solve(self.statistics(trials, exo_trials), lam, lam_exo)
