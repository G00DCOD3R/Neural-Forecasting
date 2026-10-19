# ./.venv/bin/python -m unittest models.VAR.test_varx -v
import unittest
from types import SimpleNamespace

import numpy as np

from .var import VAR1
from .varx import VARX
from .train_varx import forecast_batches_exo


class TestVARX(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.N, self.E, self.Q = 4, 2, 3
        self.xs = [rng.normal(size=(L, self.N)) + 1 for L in (30, 18, 12)]
        self.us = [rng.normal(size=(len(x), self.E)) for x in self.xs]

    def test_without_exogenous_equals_var1(self):
        empty = [np.zeros((len(x), 0)) for x in self.xs]
        model = VARX(1).fit(self.xs, empty, lam=0.13)
        reference = VAR1(0.13).fit(self.xs)
        np.testing.assert_allclose(model.coefficients, reference.coefficients, atol=1e-12)
        np.testing.assert_allclose(model.intercept, reference.intercept, atol=1e-12)

    # augmented least squares with per-block penalties, unpenalised intercept
    def test_fit_matches_augmented_least_squares(self):
        lam, lam_exo, t0 = 0.05, 0.4, self.Q - 1
        Z = np.vstack([np.c_[x[t0 - 1:-1], np.hstack([u[t0 - q:len(u) - q] for q in range(self.Q)])]
                       for x, u in zip(self.xs, self.us)])
        Y = np.vstack([x[t0:] for x in self.xs])
        M, D = Z.shape
        pen = np.sqrt(np.r_[np.full(self.N, lam), np.full(D - self.N, lam_exo)])
        A = np.vstack([np.c_[Z, np.ones(M)] / np.sqrt(M), np.c_[np.diag(pen), np.zeros(D)]])
        W = np.linalg.lstsq(A, np.vstack([Y / np.sqrt(M), np.zeros((D, self.N))]), rcond=None)[0]
        model = VARX(self.Q).fit(self.xs, self.us, lam, lam_exo)
        np.testing.assert_allclose(model.coefficients, W[:self.N], atol=1e-10)
        np.testing.assert_allclose(model.exo_coefficients, W[self.N:-1], atol=1e-10)
        np.testing.assert_allclose(model.intercept, W[-1], atol=1e-10)
        self.assertEqual(model.n_train_pairs, M)

    def test_rollout_uses_the_exogenous_rows_of_each_step(self):
        model = VARX(self.Q).fit(self.xs, self.us, 0.1, 1.0)
        x, u, t, H = self.xs[0], self.us[0], 10, 4
        pred = model.rollout(x[None, t - 1], u[None, t - self.Q + 1:t + H], H, floor=np.full(self.N, 0.5))[0]
        state = x[t - 1]
        for j in range(H):
            lags = np.concatenate([u[t + j - q] for q in range(self.Q)])
            state = np.maximum(model.predict(state[None], lags[None])[0], 0.5)
            np.testing.assert_allclose(pred[j], state)

    def test_windows_hold_behaviour_but_not_video(self):
        x, u = self.xs[0], self.us[0]
        data = SimpleNamespace(trials=lambda w: [x], exo_trials=lambda w: [u], known_ahead=np.array([True, False]))
        previous, target, uw = next(forecast_batches_exo(data, 'test', self.Q, context=5, horizon=3, batch_size=1))
        t = 5
        np.testing.assert_array_equal(previous[0], x[t - 1])
        np.testing.assert_array_equal(target[0], x[t:t + 3])
        np.testing.assert_array_equal(uw[0, :, 0], u[t - self.Q + 1:t + 3, 0])  # video: true future
        np.testing.assert_array_equal(uw[0, self.Q - 1:, 1], u[t, 1])  # behaviour: held from the first target


if __name__ == '__main__':
    unittest.main()
