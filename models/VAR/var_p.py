"""Population VAR(P) with ridge coefficients and unpenalized intercepts."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve


@dataclass(frozen=True)
class VARStatistics:
    P: int
    num_neurons: int
    count: int
    start: int
    covariance: np.ndarray
    cross_covariance: np.ndarray
    past_mean: np.ndarray
    target_mean: np.ndarray


class VAR:
    """Use chronological history [context, neurons], with context >= P.

    coefficients[lag-1, source, target] multiplies the source at that lag.
    Predictions sum the P population contributions and a per-target intercept.
    The fitting objective is ||Y-ZB-b||_F^2/M + lam*||B||_F^2.
    """

    def __init__(self, P=1, lam=0.1):
        if isinstance(P, (bool, np.bool_)) or not isinstance(P, (int, np.integer)) or P < 1:
            raise ValueError('P must be a positive integer')
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError('lam must be finite and positive')
        self.P, self.lam = int(P), float(lam)
        self.coefficients = self.intercept = None
        self.n_train_pairs = 0
        self.training_start = None

    def statistics(self, trials, start=None):
        """Accumulate centered sufficient statistics within separate trials.

        start=P uses every available training target. A common start (e.g. 15)
        gives different P values identical fitting targets. Compute once per P
        and reuse for a ridge grid; no session-sized design matrix is stacked.
        """
        start = self.P if start is None else start
        if not isinstance(start, (int, np.integer)) or start < self.P:
            raise ValueError('Training start must be an integer >= P')
        gram = cross = past_sum = target_sum = None
        count = 0
        N = None
        for trial in trials:
            x = np.asarray(trial, dtype=np.float64)
            if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
                raise ValueError('Each trial must be finite [time, neurons]')
            if N is None:
                N = x.shape[1]
                D = self.P*N
                gram, cross = np.zeros((D, D)), np.zeros((D, N))
                past_sum, target_sum = np.zeros(D), np.zeros(N)
            elif N != x.shape[1]:
                raise ValueError('Trials must have the same neuron count')
            if len(x) <= start:
                continue
            Z = np.concatenate([x[start-lag:len(x)-lag]
                                for lag in range(1, self.P+1)], axis=1)
            Y = x[start:]
            gram += Z.T @ Z
            cross += Z.T @ Y
            past_sum += Z.sum(0)
            target_sum += Y.sum(0)
            count += len(Y)
        if not count:
            raise ValueError('No training targets: a trial must be longer than training start')
        past_mean, target_mean = past_sum/count, target_sum/count
        covariance = gram/count - np.outer(past_mean, past_mean)
        cross_covariance = cross/count - np.outer(past_mean, target_mean)
        if not np.isfinite(covariance).all() or not np.isfinite(cross_covariance).all():
            raise ValueError('Nonfinite sufficient statistics')
        return VARStatistics(self.P, N, count, int(start), covariance,
                             cross_covariance, past_mean, target_mean)

    def fit_statistics(self, stats):
        """Solve a ridge fit without mutating the reusable statistics."""
        if not isinstance(stats, VARStatistics) or stats.P != self.P:
            raise ValueError('Statistics must match the model lag order')
        D = self.P*stats.num_neurons
        regularized = stats.covariance.copy()
        regularized.flat[::D+1] += self.lam
        weights = solve(regularized, stats.cross_covariance, assume_a='pos')
        self.coefficients = weights.reshape(self.P, stats.num_neurons, stats.num_neurons)
        self.intercept = stats.target_mean - stats.past_mean @ weights
        self.n_train_pairs, self.training_start = stats.count, stats.start
        return self

    def fit(self, trials, start=None):
        return self.fit_statistics(self.statistics(trials, start))

    @property
    def parameter_count(self):
        if self.coefficients is None:
            raise RuntimeError('Fit the model first')
        return self.coefficients.size + self.intercept.size

    def _history(self, history):
        if self.coefficients is None:
            raise RuntimeError('Fit the model before predicting')
        x = np.asarray(history, dtype=np.float64)
        if (x.ndim not in (2, 3) or x.shape[-2] < self.P
                or x.shape[-1] != len(self.intercept) or not np.isfinite(x).all()):
            raise ValueError(f'Expected finite [context, neurons] or [batch, context, neurons], context >= {self.P}')
        return x[..., -self.P:, :]

    def predict(self, history):
        x = self._history(history)
        features = x[..., ::-1, :].reshape(-1, self.P*len(self.intercept))
        result = features @ self.coefficients.reshape(self.P*len(self.intercept), -1) + self.intercept
        return result[0] if x.ndim == 2 else result

    def rollout(self, history, horizon=5):
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        state = self._history(history).copy()
        predictions = []
        for _ in range(horizon):
            with np.errstate(over='raise', invalid='raise'):
                prediction = self.predict(state)
            if not np.isfinite(prediction).all():
                raise FloatingPointError('Recursive forecast became nonfinite')
            predictions.append(prediction)
            state = np.concatenate([state[..., 1:, :], prediction[..., None, :]], axis=-2)
        return np.stack(predictions, axis=-2)
