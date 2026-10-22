"""Independent numerical and protocol checks for multi-lag VAR selection."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .var import VAR1
from .var_p import VAR
from .lag_selection import evaluate, history_batches, select_lags
from .select_lags import run
from .data import load_session
from .data_example import write_synthetic_session


class TestMultiLagVAR(unittest.TestCase):
    def test_two_lag_fit_matches_independent_augmented_least_squares(self):
        rng = np.random.default_rng(11)
        trials = [rng.normal(size=(31, 3))+2, rng.normal(size=(23, 3))-1]
        P, start, lam = 2, 4, 0.17
        Z = np.array([np.concatenate([x[t-1], x[t-2]]) for x in trials for t in range(start, len(x))])
        Y = np.array([x[t] for x in trials for t in range(start, len(x))])
        M, D = Z.shape
        design = np.column_stack([Z, np.ones(M)])/np.sqrt(M)
        penalty = np.column_stack([np.sqrt(lam)*np.eye(D), np.zeros(D)])
        expected = np.linalg.lstsq(np.vstack([design, penalty]),
                                  np.vstack([Y/np.sqrt(M), np.zeros((D, 3))]), rcond=None)[0]
        model = VAR(P, lam).fit(trials, start=start)
        np.testing.assert_allclose(model.coefficients.reshape(D, 3), expected[:-1], atol=1e-12)
        np.testing.assert_allclose(model.intercept, expected[-1], atol=1e-12)
        histories = np.array([x[t-P:t] for x in trials for t in range(start, len(x))])
        np.testing.assert_allclose(model.predict(histories), Z@expected[:-1]+expected[-1], atol=1e-12)
        self.assertEqual(model.n_train_pairs, sum(len(x)-start for x in trials))
        self.assertEqual(model.parameter_count, P*3*3+3)

    def test_one_lag_reproduces_original_with_identical_training_targets(self):
        rng = np.random.default_rng(7)
        trials = [rng.normal(size=(30, 3)), rng.normal(size=(25, 3))]
        old = VAR1(0.1).fit(trials)
        new = VAR(1, 0.1).fit(trials)
        np.testing.assert_allclose(new.coefficients[0], old.coefficients, atol=1e-12)
        np.testing.assert_allclose(new.intercept, old.intercept, atol=1e-12)
        np.testing.assert_allclose(new.predict(trials[0][:15]), old.predict(trials[0][14]), atol=1e-12)
        np.testing.assert_allclose(new.rollout(trials[0][:15], 5), old.rollout(trials[0][14], 5), atol=1e-12)
        self.assertEqual(new.n_train_pairs, old.n_train_pairs)

    def test_two_lag_recursive_forecast_matches_hand_calculation(self):
        model = VAR(2)
        model.coefficients = np.array([[[0.5]], [[0.25]]])
        model.intercept = np.array([1.0])
        history = np.array([[2.0], [4.0]])
        copy = history.copy()
        np.testing.assert_allclose(model.rollout(history, 3), [[3.5], [3.75], [3.75]])
        np.testing.assert_allclose(model.rollout(history[None], 3), [[[3.5], [3.75], [3.75]]])
        np.testing.assert_array_equal(history, copy)

    def test_common_target_rows_and_statistics_stay_inside_trials(self):
        trials = [np.arange(18.).reshape(9, 2), 100+np.arange(16.).reshape(8, 2)]
        targets = []
        for P in (1, 2, 4):
            model = VAR(P)
            st = model.statistics(trials, start=4)
            self.assertEqual(st.count, 9)
            batches = list(history_batches(trials, P, context=4, horizon=2, batch_size=2))
            actual = np.concatenate([y for _, y in batches])
            expected = np.array([x[t:t+2] for x in trials for t in range(4, len(x)-1)])
            np.testing.assert_array_equal(actual, expected)
            expected_history = np.array([x[t-P:t] for x in trials for t in range(4, len(x)-1)])
            np.testing.assert_array_equal(np.concatenate([x for x, _ in batches]), expected_history)
            targets.append(actual)
        np.testing.assert_array_equal(targets[0], targets[-1])

    def test_statistics_reuse_and_validation_only_selection(self):
        rng = np.random.default_rng(8)
        train, val = [rng.normal(size=(25, 3))], [rng.normal(size=(21, 3))]
        stats = VAR(2).statistics(train, start=4)
        before = stats.covariance.copy()
        a = VAR(2, 0.1).fit_statistics(stats)
        VAR(2, 1).fit_statistics(stats)
        np.testing.assert_array_equal(stats.covariance, before)
        np.testing.assert_array_equal(a.coefficients, VAR(2, 0.1).fit(train, start=4).coefficients)
        model, grid = select_lags(iter(train), iter(val), [1, 2], [0.01, 0.1], context=4, verbose=False)
        best = min(grid, key=lambda row: row['validation']['mse'][0])
        self.assertEqual((model.P, model.lam), (best['P'], best['lambda']))
        self.assertEqual({r['fitted_training_pairs'] for r in grid}, {21})
        self.assertEqual({r['validation']['n_windows'] for r in grid}, {17})
        expected = VAR(model.P, model.lam).fit(train, start=4)
        np.testing.assert_array_equal(model.coefficients, expected.coefficients)

    def test_metrics_match_direct_common_definitions(self):
        rng = np.random.default_rng(19)
        trials = [rng.normal(size=(18, 3)), rng.normal(size=(24, 3))]
        for x in trials:
            x[:, 2] = 0
        model = VAR(2, 0.1).fit(trials, start=4)
        result = evaluate(model, trials, context=4, horizon=3, batch_size=2)
        batches = list(history_batches(trials, 2, context=4, horizon=3))
        prediction = np.concatenate([model.rollout(h, 3) for h, _ in batches])
        target = np.concatenate([y for _, y in batches])
        error = prediction-target
        pc, tc = prediction-prediction.mean(0), target-target.mean(0)
        mse = (error**2).mean((0, 2))
        pearson = ((pc*tc).sum(0)/(np.linalg.norm(pc, axis=0)*np.linalg.norm(tc, axis=0)+1e-8)).mean(-1)
        r2 = (1-(error**2).sum(0)/((tc**2).sum(0)+1e-8)).mean(-1)
        np.testing.assert_allclose(result['mse'], mse, atol=1e-12)
        np.testing.assert_allclose(result['rmse'], np.sqrt(mse), atol=1e-12)
        np.testing.assert_allclose(result['mae'], abs(error).mean((0, 2)), atol=1e-12)
        np.testing.assert_allclose(result['pearson'], pearson, atol=1e-12)
        np.testing.assert_allclose(result['r2'], r2, atol=1e-12)

    def test_rejects_invalid_lags_context_and_histories(self):
        for P in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                VAR(P)
        with self.assertRaises(ValueError):
            VAR(2).fit([np.zeros((8, 3))], start=1)
        with self.assertRaises(ValueError):
            VAR(2).fit([np.zeros((2, 3))])
        with self.assertRaises(ValueError):
            select_lags([np.zeros((8, 3))], [np.zeros((8, 3))], lags=[4], context=3, verbose=False)
        model = VAR(2).fit([np.zeros((8, 3))])
        with self.assertRaises(ValueError):
            model.predict(np.zeros((1, 3)))

    def test_saved_selection_and_test_data_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root/'session'
            write_synthetic_session(data_dir, length=40)
            first, result, output = run(data_dir, lags=[1, 2], lambdas=[0.01, 0.1],
                                        context=4, output_dir=root/'first', verbose=False)
            self.assertNotIn('test', result)
            self.assertEqual(json.loads((output/'selection.json').read_text()), result)
            data = load_session(data_dir)
            with np.load(output/'model.npz', allow_pickle=False) as saved:
                restored = VAR(int(saved['P']), float(saved['lam']))
                restored.coefficients, restored.intercept = saved['coefficients'], saved['intercept']
                np.testing.assert_array_equal(saved['mu'], data.mu)
                np.testing.assert_array_equal(restored.predict(data.trials('val')[0][:4]),
                                              first.predict(data.trials('val')[0][:4]))
            with self.assertRaises(FileExistsError):
                run(data_dir, output_dir=output, verbose=False)
            X = np.load(data_dir/'X_features.npy')
            X[17*40:] += 1000
            np.save(data_dir/'X_features.npy', X)
            second, other, _ = run(data_dir, lags=[1, 2], lambdas=[0.01, 0.1],
                                   context=4, output_dir=root/'second', verbose=False)
            self.assertEqual(result['selected'], other['selected'])
            self.assertEqual(result['grid'], other['grid'])
            np.testing.assert_array_equal(first.coefficients, second.coefficients)


if __name__ == '__main__':
    unittest.main()
