"""Reduced-rank ridge VAR using a supervised, training-only output subspace.

Reference: Mukherjee & Zhu (2011), doi:10.1002/sam.10138.
For centered statistics C=Z.T@Z/M, D=Z.T@Y/M and G=C+lambda*I,
B_r=G^-1 D V_r V_r.T, where V_r are top eigenvectors of D.T G^-1 D.
This minimizes ||Y-ZB-b||_F^2/M + lambda*||B||_F^2, rank(B)<=r.
The rank constraint applies to the stacked [P*N,N] matrix across all lags.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

try:
    from .var_p import VAR, VARStatistics
except ImportError:
    from var_p import VAR, VARStatistics


@dataclass(frozen=True)
class RankProjection:
    """One full ridge fit and eigendecomposition, reusable across rank values."""
    full_model: VAR
    output_basis: np.ndarray
    past_mean: np.ndarray
    target_mean: np.ndarray

    @classmethod
    def from_statistics(cls, stats, lam):
        if not isinstance(stats, VARStatistics):
            raise ValueError('Expected VARStatistics')
        full = VAR(stats.P, lam).fit_statistics(stats)
        weights = full.coefficients.reshape(stats.P*stats.num_neurons, stats.num_neurons)
        energy = stats.cross_covariance.T @ weights
        _, vectors = eigh((energy+energy.T)/2)
        return cls(full, vectors[:, ::-1], stats.past_mean.copy(), stats.target_mean.copy())

    def model(self, rank):
        N = self.output_basis.shape[0]
        if (isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer))
                or not 1 <= rank <= N):
            raise ValueError(f'Rank must be an integer from 1 to {N}')
        model = ReducedRankVAR(self.full_model.P, rank, self.full_model.lam)
        model.output_factors = self.output_basis[:, :rank].copy()
        weights = self.full_model.coefficients.reshape(self.full_model.P*N, N)
        model.input_factors = weights @ model.output_factors
        model.intercept = self.target_mean - (self.past_mean @ model.input_factors) @ model.output_factors.T
        model.n_train_pairs = self.full_model.n_train_pairs
        model.training_start = self.full_model.training_start
        return model


class ReducedRankVAR:
    """Factored VAR(P): predict = features @ input_factors @ output_factors.T + b.

    Predictions and checkpoints use the factors. The dense coefficients property
    materializes [P,N,N] only when explicitly requested for diagnostics.
    parameter_count counts stored factor/intercept values, including the factor
    rotation redundancy. coefficient_dimension reports r*(P*N+N-r)+N: the maximum
    rank-r coefficient-manifold dimension, not statistical effective degrees of freedom.
    """
    def __init__(self, P=2, rank=64, lam=0.1):
        base = VAR(P, lam)
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or rank < 1:
            raise ValueError('Rank must be a positive integer')
        self.P, self.lam, self.rank = base.P, base.lam, int(rank)
        self.input_factors = self.output_factors = self.intercept = None
        self.n_train_pairs = 0
        self.training_start = None

    def fit(self, trials, start=None):
        stats = VAR(self.P, self.lam).statistics(trials, start)
        if self.rank > stats.num_neurons:
            raise ValueError('Rank exceeds the number of output neurons')
        fitted = RankProjection.from_statistics(stats, self.lam).model(self.rank)
        self.input_factors, self.output_factors = fitted.input_factors, fitted.output_factors
        self.intercept = fitted.intercept
        self.n_train_pairs, self.training_start = fitted.n_train_pairs, fitted.training_start
        return self

    def _fitted(self):
        if self.input_factors is None or self.output_factors is None or self.intercept is None:
            raise RuntimeError('Fit the model before using it')

    @property
    def coefficients(self):
        self._fitted()
        return (self.input_factors @ self.output_factors.T).reshape(self.P, len(self.intercept), -1)

    @property
    def parameter_count(self):
        self._fitted()
        return self.input_factors.size+self.output_factors.size+self.intercept.size

    @property
    def coefficient_dimension(self):
        self._fitted()
        N = len(self.intercept)
        return self.rank*(self.P*N+N-self.rank)+N

    def _history(self, history):
        self._fitted()
        x = np.asarray(history, dtype=np.float64)
        if (x.ndim not in (2, 3) or x.shape[-2] < self.P or x.shape[-1] != len(self.intercept)
                or not np.isfinite(x).all()):
            raise ValueError(f'Expected finite chronological history with context >= {self.P}')
        return x[..., -self.P:, :]

    def predict(self, history):
        x = self._history(history)
        features = x[..., ::-1, :].reshape(-1, self.P*len(self.intercept))
        prediction = (features @ self.input_factors) @ self.output_factors.T + self.intercept
        return prediction[0] if x.ndim == 2 else prediction

    # The existing recursive-history update needs only _history() and predict().
    rollout = VAR.rollout
