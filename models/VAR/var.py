"""One-lag population VAR with an unpenalized intercept and ridge fitting."""
import numpy as np
from scipy.linalg import solve


class VAR1:
    """Predict x[t] from the population vector x[t-1].

    We use row vectors: prediction = previous @ coefficients + intercept.
    coefficients[source, target] describes source's contribution to target.
    """

    def __init__(self, lam=0.1):
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError('lam must be finite and positive')
        self.lam = float(lam)
        self.coefficients = None
        self.intercept = None
        self.n_train_pairs = 0

    def fit(self, trials):
        """Fit separate [time, neurons] trials without crossing their boundaries.

        Minimize ||Y - Z B - b||_F^2 / M + lam * ||B||_F^2,
        where M is the number of within-trial training pairs. Only B is penalized.
        Accumulating sufficient statistics avoids stacking all training pairs.
        """
        gram = cross = z_sum = y_sum = None
        count = 0
        N = None
        for trial in trials:
            x = np.asarray(trial, dtype=np.float64)
            if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
                raise ValueError('Each trial must be a finite [time, neurons] array')
            if N is None:
                N = x.shape[1]
                gram = np.zeros((N, N))
                cross = np.zeros((N, N))
                z_sum, y_sum = np.zeros(N), np.zeros(N)
            elif x.shape[1] != N:
                raise ValueError('All trials must have the same neuron count')
            if len(x) < 2:
                continue

            # Adjacent rows belong to the same trial, by construction.
            Z, Y = x[:-1], x[1:]
            gram += Z.T @ Z
            cross += Z.T @ Y
            z_sum += Z.sum(axis=0)
            y_sum += Y.sum(axis=0)
            count += len(Z)

        if not count:
            raise ValueError('At least one training trial must contain two timesteps')
        z_mean, y_mean = z_sum / count, y_sum / count
        # Center the normal equations so the intercept is not regularized.
        gram = gram / count - np.outer(z_mean, z_mean)
        cross = cross / count - np.outer(z_mean, y_mean)
        gram.flat[::N + 1] += self.lam
        coefficients = solve(gram, cross, assume_a='pos')
        intercept = y_mean - z_mean @ coefficients

        self.coefficients = coefficients
        self.intercept = intercept
        self.n_train_pairs = count
        return self

    @property
    def parameter_count(self):
        if self.coefficients is None:
            raise RuntimeError('Fit the model first')
        return self.coefficients.size + self.intercept.size

    def predict(self, previous):
        """Accept one [neurons] vector or a [samples, neurons] batch."""
        if self.coefficients is None:
            raise RuntimeError('Fit the model before predicting')
        x = np.asarray(previous, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.intercept):
            raise ValueError(f'Expected {len(self.intercept)} neurons in the last dimension')
        if not np.isfinite(x).all():
            raise ValueError('Prediction inputs must be finite')
        return x @ self.coefficients + self.intercept

    def rollout(self, previous, horizon=5):
        """Forecast recursively from the latest observed population vector.

        [neurons] -> [horizon, neurons]; [batch, neurons] -> [batch, horizon, neurons].
        Predictions, rather than future observations, become the next input.
        """
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        state = np.asarray(previous, dtype=np.float64).copy()
        predictions = []
        for _ in range(horizon):
            with np.errstate(over='raise', invalid='raise'):
                state = self.predict(state)
            if not np.isfinite(state).all():
                raise FloatingPointError('Recursive forecast became nonfinite')
            predictions.append(state)
        return np.stack(predictions, axis=-2)
