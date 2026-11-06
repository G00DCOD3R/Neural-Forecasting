# VAR-X: one-lag population VAR with lagged exogenous inputs, ridge fitted in closed form.
#   prediction = previous @ coefficients + [u_t, ..., u_{t-Q+1}] @ exo_coefficients + intercept
# Separate penalties for the activity (lam) and exogenous (lam_exo) blocks, intercept unpenalised.
# Without exogenous inputs this is VAR1.
import numpy as np
from scipy.linalg import solve


class VARX:
    def __init__(self, Q=1):
        if not isinstance(Q, (int, np.integer)) or Q < 1:
            raise ValueError('Q must be a positive integer')
        self.Q = int(Q)
        self.lam = self.lam_exo = None
        self.coefficients = self.exo_coefficients = self.intercept = None
        self.n_train_pairs = 0

    @property
    def parameter_count(self):
        return self.coefficients.size + self.exo_coefficients.size + self.intercept.size

    # [L, E] -> [L - t0, Q*E] for targets t0..L-1, row t = [u_t, ..., u_{t-Q+1}]
    def lagged(self, u, t0):
        return np.concatenate([u[t0 - q:len(u) - q] for q in range(self.Q)], 1)

    # Sums for the normal equations over [time, neurons] trials and their [time, E] inputs
    def statistics(self, trials, exo_trials):
        t0 = max(1, self.Q - 1)
        gram = cross = z_sum = y_sum = None
        count = 0
        for x, u in zip(trials, exo_trials):
            x, u = np.asarray(x, np.float64), np.asarray(u, np.float64)
            if len(x) != len(u) or not np.isfinite(x).all() or not np.isfinite(u).all():
                raise ValueError('Each trial needs finite activity and exogenous rows of equal length')
            if len(x) <= t0:
                continue
            Z = np.concatenate([x[t0 - 1:-1], self.lagged(u, t0)], 1)  # within one trial by construction
            Y = x[t0:]
            if gram is None:
                gram, cross = np.zeros((Z.shape[1],) * 2), np.zeros((Z.shape[1], Y.shape[1]))
                z_sum, y_sum = np.zeros(Z.shape[1]), np.zeros(Y.shape[1])
            gram += Z.T @ Z
            cross += Z.T @ Y
            z_sum += Z.sum(0)
            y_sum += Y.sum(0)
            count += len(Z)
        if not count:
            raise ValueError('No training pairs: trials are shorter than the lags')
        return {'gram': gram, 'cross': cross, 'z_sum': z_sum, 'y_sum': y_sum, 'count': count,
                'N': cross.shape[1]}

    # Minimise ||Y - Z W - b||^2 / M + lam ||A||^2 + lam_exo ||B||^2, centred so b is unpenalised
    def solve(self, st, lam=0.1, lam_exo=1.0):
        if any(not np.isfinite(v) or v <= 0 for v in (lam, lam_exo)):
            raise ValueError('Penalties must be finite and positive')
        n, N = st['count'], st['N']
        z_mean, y_mean = st['z_sum'] / n, st['y_sum'] / n
        gram = st['gram'] / n - np.outer(z_mean, z_mean)
        cross = st['cross'] / n - np.outer(z_mean, y_mean)
        D = len(gram)
        gram[np.arange(D), np.arange(D)] += np.r_[np.full(N, lam), np.full(D - N, lam_exo)]
        W = solve(gram, cross, assume_a='pos')
        self.lam, self.lam_exo = float(lam), float(lam_exo)
        self.coefficients, self.exo_coefficients = W[:N], W[N:]
        self.intercept = y_mean - z_mean @ W
        self.n_train_pairs = n
        return self

    def fit(self, trials, exo_trials, lam=0.1, lam_exo=1.0):
        trials, exo_trials = list(trials), list(exo_trials)
        return self.solve(self.statistics(trials, exo_trials), lam, lam_exo)

    # previous [batch, N], u_lags [batch, Q*E] -> [batch, N]
    def predict(self, previous, u_lags):
        if self.coefficients is None:
            raise RuntimeError('Fit the model before predicting')
        return previous @ self.coefficients + u_lags @ self.exo_coefficients + self.intercept

    # Recursive forecast [batch, horizon, N] from the last observation [batch, N].
    # u [batch, Q-1+horizon, E]: rows u_{t-Q+1}, ..., u_{t+horizon-1} for the first target t.
    # With floor [N], each prediction is clamped at it before it is fed back.
    def rollout(self, previous, u, horizon=5, floor=None):
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        u = np.asarray(u, np.float64)
        if u.shape[1] != self.Q - 1 + horizon:
            raise ValueError(f'Expected {self.Q - 1 + horizon} exogenous rows per window')
        state = np.asarray(previous, np.float64).copy()
        predictions = []
        for j in range(horizon):
            lags = u[:, j:j + self.Q][:, ::-1].reshape(len(u), -1)  # [u_t, u_{t-1}, ...]
            with np.errstate(over='raise', invalid='raise'):
                state = self.predict(state, lags)
            if floor is not None:
                state = np.maximum(state, floor)
            predictions.append(state)
        return np.stack(predictions, axis=-2)
