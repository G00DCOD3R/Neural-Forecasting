# Direct multi-horizon G-VAR: one G-VAR head per horizon, all forecasting from the same observed context.
#   x_{t+j} = G-VAR_j(x_{t-1}, ..., x_{t-P}, u),  j = 0..H-1
# Nothing is fed back, so errors cannot compound. Each head is fitted in closed form.
import torch
import torch.nn as nn

from gvar import GVAR


class DirectGVAR(nn.Module):
    def __init__(self, S, P, K, H=5, leaky=False, n_exo=0, Q=1, intercept=False, known_ahead=None):
        super().__init__()
        self.heads = nn.ModuleList(GVAR(S, P, K, leaky, n_exo, Q, intercept) for _ in range(H))
        self.known_ahead = known_ahead
        self.floor = None  # [N], if set forecasts are clamped at it

    S = property(lambda self: self.heads[0].S)
    P = property(lambda self: self.heads[0].P)
    K = property(lambda self: self.heads[0].K)
    Q = property(lambda self: self.heads[0].Q)
    n_exo = property(lambda self: self.heads[0].n_exo)
    c = property(lambda self: self.heads[0].c)
    H = property(lambda self: len(self.heads))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    # Head j regresses x_{t+j} on the context, with behaviour held at the origin
    def statistics(self, trials, exo_trials=None):
        return [h.statistics(trials, exo_trials, j, self.known_ahead) for j, h in enumerate(self.heads)]

    def solve(self, sts, lam=1e-3, lam_exo=1.0):
        for h, st in zip(self.heads, sts):
            h.solve(st, lam, lam_exo)
        return self

    # x_past [B, P, N], u [B, Q-1+horizon, E] (rows u_{t-Q+1..t+horizon-1}) -> [B, horizon, N]
    def forward(self, x_past, u=None, horizon=None):
        horizon = horizon or len(self.heads)
        return torch.stack([h(x_past, u[:, j:j + self.Q] if self.n_exo else None)
                            for j, h in enumerate(self.heads[:horizon])], 1)

    # Same interface as GVAR.rollout, but every horizon comes straight from the context
    @torch.no_grad()
    def rollout(self, x_past, horizon, u=None):
        assert horizon <= self.H, f"model has {self.H} horizons"
        pred = self(x_past, u, horizon)
        return pred if self.floor is None else torch.maximum(pred, self.floor)
