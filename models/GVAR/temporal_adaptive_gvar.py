"""Activity-dependent Adaptive-GVAR: shared GRU, Q/K attention, linear GVAR."""
import math

import torch
from torch import nn

from gvar import GVAR


def sparsemax(scores):
    """Project each last-dimension row onto the probability simplex.

    Sorting/thresholding implements Martins & Astudillo (2016), Algorithm 1.
    Autograd differentiates through the selected support (away from boundaries).
    Scores and weights remain dense tensors; exact zeros do not imply a speedup.
    """
    if scores.ndim < 1 or scores.shape[-1] == 0 or not scores.is_floating_point():
        raise ValueError('Expected floating-point scores with nonempty last dimension')
    dtype = scores.dtype
    z = scores.float() if dtype in (torch.float16, torch.bfloat16) else scores
    z = z - z.amax(dim=-1, keepdim=True)
    # With max(z)=0 the simplex threshold is in [-1, 0]. Values below -1
    # cannot be active; clipping them prevents overflow in cumulative sums.
    z = z.clamp_min(-1)
    ordered = z.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1)
    ranks = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    support_size = (1 + ranks * ordered > cumulative).sum(dim=-1, keepdim=True)
    threshold = (cumulative.gather(-1, support_size - 1) - 1) / support_size
    return (z - threshold).clamp_min(0).to(dtype)


@torch.no_grad()
def correction_statistics(C):
    """Exact support statistics over all window/target rows, including self-inputs."""
    counts = (C > 0).sum(dim=-1)
    return {
        'zero_fraction': (C == 0).float().mean().item(),
        'mean_nonzero_incoming_edges': counts.float().mean().item(),
        'singleton_row_fraction': (counts == 1).float().mean().item(),
    }


class TemporalGraphLearner(nn.Module):
    """[B, P, N] histories -> [B, N, N] corrections (target rows, source columns).

    All neurons use the same encoder and projections. No neuron embeddings,
    candidate masks or neuron-specific parameters; self-connections are allowed.
    """

    def __init__(self, hidden_dim=16, activation='softmax', tau=1.0):
        super().__init__()
        if activation not in ('softmax', 'sparsemax'):
            raise ValueError('activation must be softmax or sparsemax')
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('tau must be positive and finite')
        self.encoder = nn.GRU(1, hidden_dim, num_layers=1, batch_first=True)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = activation
        self.tau = float(tau)  # Fixed configuration, not a learned parameter.

    def normalize_scores(self, scores):
        """Apply fixed temperature and the configured row-wise normalization."""
        # Subtracting a row constant leaves either normalization unchanged and
        # avoids positive overflow when a small temperature sharpens the scores.
        scaled = (scores - scores.amax(dim=-1, keepdim=True)) / self.tau
        return scaled.softmax(dim=-1) if self.activation == 'softmax' else sparsemax(scaled)

    def forward(self, past):
        batch, history, nodes = past.shape
        x = past.transpose(1, 2).reshape(batch * nodes, history, 1)
        _, hidden = self.encoder(x)  # Fresh hidden state for each neuron/window.
        H = hidden[0].reshape(batch, nodes, -1)
        Q, K = self.W_q(H), self.W_k(H)
        Z = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
        return self.normalize_scores(Z)


class AdaptiveGVAR(GVAR):
    """Original GVAR equation with a graph inferred separately for each window.

    ``S`` stores the fixed anatomical shift for compatibility with GVAR utilities;
    ``graph_shift(past)`` returns the actual mixed shift used for that window.
    Forecasting retains the original activity values as its temporal inputs.
    Inherited rollout recomputes the graph after each autoregressive step.
    """

    def __init__(self, A, P=15, K=3, alpha=0.1, hidden_dim=16, activation='softmax', tau=1.0):
        if P < 1 or K < 1 or not 0 <= alpha < 1:
            raise ValueError('Require P >= 1, K >= 1 and 0 <= alpha < 1')
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('Expected a square anatomical shift [N, N]')
        # Reuse data.shift_operator(..., kind="mean"): target rows, source columns.
        A = A.detach().to_dense().clone()
        if not torch.isfinite(A).all() or (A < 0).any():
            raise ValueError('Anatomical weights must be finite and nonnegative')
        row_sums = A.sum(-1)
        if not torch.all(torch.isclose(row_sums, torch.ones_like(row_sums)) | (row_sums == 0)):
            raise ValueError('Anatomical rows must sum to one or be empty')
        super().__init__(A, P, K)
        del self.S
        self.register_buffer('S', A)
        self.graph_learner = TemporalGraphLearner(hidden_dim, activation, tau)
        self.alpha = float(alpha)  # Fixed hyperparameter, not a learned parameter.
        # Zero h would block gradients into the graph learner on the first update.
        nn.init.normal_(self.h, mean=0.0, std=0.01)

    @property
    def A(self):
        return self.S

    def graph_shift(self, past):
        """[B, P, N] -> [B, N, N]; do not renormalize the convex mixture."""
        if self.alpha == 0:
            return self.A.unsqueeze(0).expand(past.shape[0], -1, -1)
        C = self.graph_learner(past)
        return (1 - self.alpha) * self.A + self.alpha * C

    def features(self, x_past):
        """Same lag order/features as GVAR, using one graph per batch element."""
        if x_past.ndim != 3 or x_past.shape[1:] != (self.P, self.A.shape[0]):
            raise ValueError(f'Expected past [B, {self.P}, {self.A.shape[0]}]')
        S = self.graph_shift(x_past)
        shifted = x_past.flip(1).transpose(1, 2)  # [B, N, P], lag 1 first.
        features = [shifted]
        for _ in range(self.K):
            shifted = S @ shifted
            features.append(shifted)
        return torch.stack(features, dim=2).unsqueeze(2)  # [B, N, 1, K+1, P]

    def fit(self, *args, **kwargs):
        raise NotImplementedError('Train jointly with forecasting MSE; see adaptive_gvar.ipynb.')
