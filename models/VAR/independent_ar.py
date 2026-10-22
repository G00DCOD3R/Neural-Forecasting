"""Independent per-neuron AR(P) control, using the same ridge convention as VAR."""
import numpy as np

try:
    from .var_p import VAR, VARStatistics
except ImportError:
    from var_p import VAR, VARStatistics


class IndependentAR:
    """Predict each target from its own P lags and an unpenalized intercept.

    lag_weights[p-1, n] multiplies neuron n's activity at lag p. This is the
    diagonal restriction of population VAR(P), with the same per-time ridge
    objective and fitting targets. No neuron uses another neuron's activity.
    """
    def __init__(self, P=2, lam=0.1):
        base = VAR(P, lam)
        self.P, self.lam = base.P, base.lam
        self.lag_weights = self.intercept = None
        self.n_train_pairs = 0
        self.training_start = None

    def fit_statistics(self, stats):
        if not isinstance(stats, VARStatistics) or stats.P != self.P:
            raise ValueError('Statistics must match the model lag order')
        N = stats.num_neurons
        index = np.arange(self.P)[:, None]*N + np.arange(N)[None, :]
        gram = stats.covariance[index[:, None, :], index[None, :, :]].transpose(2, 0, 1).copy()
        cross = stats.cross_covariance[index, np.arange(N)[None, :]].T
        gram += self.lam*np.eye(self.P)
        self.lag_weights = np.linalg.solve(gram, cross[..., None])[..., 0].T
        self.intercept = stats.target_mean - (stats.past_mean.reshape(self.P, N)*self.lag_weights).sum(0)
        self.n_train_pairs, self.training_start = stats.count, stats.start
        return self

    def fit(self, trials, start=None):
        # Convenience API reuses the audited population sufficient statistics.
        # The comparison runner computes those once for both model families.
        return self.fit_statistics(VAR(self.P).statistics(trials, start))

    @property
    def parameter_count(self):
        if self.lag_weights is None:
            raise RuntimeError('Fit the model first')
        return self.lag_weights.size+self.intercept.size

    def _history(self, history):
        if self.lag_weights is None or self.intercept is None:
            raise RuntimeError('Fit before predicting')
        x = np.asarray(history, dtype=np.float64)
        if (x.ndim not in (2, 3) or x.shape[-2] < self.P
                or x.shape[-1] != len(self.intercept) or not np.isfinite(x).all()):
            raise ValueError(f'Expected finite chronological history with context >= {self.P}')
        return x[..., -self.P:, :]

    def predict(self, history):
        x = self._history(history)
        return (x[..., ::-1, :]*self.lag_weights).sum(axis=-2)+self.intercept

    rollout = VAR.rollout
