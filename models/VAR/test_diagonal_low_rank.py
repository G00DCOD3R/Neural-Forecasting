"""Independent optimization, forecasting and protocol checks for factored VAR."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .compare_controls import run as run_controls
from .data import load_session
from .data_example import write_synthetic_session
from .diagonal_low_rank import DiagonalLowRankVAR, target_energy
from .independent_ar import IndependentAR
from .prediction_bounds import BoundedForecast
from .select_diagonal_rank import load_checkpoint, run, select_diagonal_rank
from .var_p import VAR


def design(trials, P=2, start=4):
    Z = np.array([np.concatenate([x[t-lag] for lag in range(1, P+1)])
                  for x in trials for t in range(start, len(x))])
    Y = np.array([x[t] for x in trials for t in range(start, len(x))])
    return Z, Y


class TestDiagonalLowRankVAR(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(67)
        self.train = [rng.normal(size=(43, 3))+2, rng.normal(size=(37, 3))-1]
        self.val = [rng.normal(size=(24, 3))+0.5]

    def test_rank_zero_and_iteration_zero_reproduce_independent_ar(self):
        independent = IndependentAR(2, 0.03).fit(self.train, start=4)
        for rank, max_iter in [(0, 20), (2, 0)]:
            model = DiagonalLowRankVAR(2, rank, 0.03).fit(self.train, start=4, max_iter=max_iter)
            np.testing.assert_array_equal(model.lag_weights, independent.lag_weights)
            np.testing.assert_array_equal(model.intercept, independent.intercept)
            np.testing.assert_array_equal(model.rollout(self.val[0][:4], 5),
                                          independent.rollout(self.val[0][:4], 5))
            self.assertEqual(model.n_iterations, 0)
            self.assertEqual(model.n_train_pairs, 72)

    def test_first_conditional_updates_match_independent_augmented_least_squares(self):
        Z, Y = design(self.train)
        Zc, Yc = Z-Z.mean(0), Y-Y.mean(0)
        M, D = Z.shape
        lam_self, lam_population, rank = 0.03, 0.2, 1
        independent = IndependentAR(2, lam_self).fit(self.train, start=4)
        E = np.zeros((2, 3, 3))
        E[:, np.arange(3), np.arange(3)] = independent.lag_weights
        residual = Yc-Zc@E.reshape(D, 3)
        augmented_x = np.vstack([Zc/np.sqrt(M), np.sqrt(lam_population)*np.eye(D)])
        augmented_y = np.vstack([residual/np.sqrt(M), np.zeros((D, 3))])
        Q, R = np.linalg.qr(augmented_x, mode='reduced')
        U, s, Vt = np.linalg.svd(Q.T@augmented_y, full_matrices=False)
        expected_population = np.linalg.solve(R, (U[:, :rank]*s[:rank])@Vt[:rank])
        model = DiagonalLowRankVAR(2, rank, lam_self, lam_population).fit(
            self.train, start=4, max_iter=1, tol=0)
        np.testing.assert_allclose(model.input_factors@model.output_factors.T,
                                   expected_population, atol=1e-11)
        residual = Yc-Zc@expected_population
        for n in range(3):
            own_z = Zc[:, [n, n+3]]
            augmented = np.vstack([own_z/np.sqrt(M), np.sqrt(lam_self)*np.eye(2)])
            expected = np.linalg.lstsq(augmented, np.r_[residual[:, n]/np.sqrt(M), np.zeros(2)], rcond=None)[0]
            np.testing.assert_allclose(model.lag_weights[:, n], expected, atol=1e-11)
        np.testing.assert_allclose(model.intercept, Y.mean(0)-Z.mean(0)@model.coefficients.reshape(D, 3), atol=1e-12)

    def test_objective_matches_direct_data_calculation_and_never_increases(self):
        Z, Y = design(self.train)
        observed = []

        def monitor(model, iteration, objective):
            B = model.coefficients.reshape(6, 3)
            population = model.input_factors@model.output_factors.T
            direct = np.square(Y-Z@B-model.intercept).sum()/len(Y)
            direct += model.lam_self*np.square(model.lag_weights).sum()
            direct += model.lam_population*np.square(population).sum()
            np.testing.assert_allclose(objective, direct/3, atol=1e-12)
            observed.append(objective)

        stats = VAR(2).statistics(self.train, start=4)
        before = stats.covariance.copy(), stats.cross_covariance.copy()
        DiagonalLowRankVAR(2, 1).fit_statistics(stats, target_energy(self.train, stats),
                                               max_iter=8, tol=0, on_iteration=monitor)
        self.assertGreater(len(observed), 1)
        self.assertTrue(np.all(np.diff(observed) <= 1e-12))
        np.testing.assert_array_equal(stats.covariance, before[0])
        np.testing.assert_array_equal(stats.cross_covariance, before[1])

    def test_full_population_rank_converges_to_independent_joint_ridge_solution(self):
        Z, Y = design(self.train)
        Zc, Yc = Z-Z.mean(0), Y-Y.mean(0)
        M, D = Z.shape
        model = DiagonalLowRankVAR(2, 3, 0.3, 0.5).fit(self.train, start=4, max_iter=200, tol=1e-14)
        expected_population, expected_self = np.zeros((D, 3)), np.zeros((2, 3))
        for n in range(3):
            # Full population and own-history predictors overlap intentionally;
            # use an independent augmented solve with separate block penalties.
            combined = np.column_stack([Zc, Zc[:, [n, n+3]]])
            penalty = np.diag(np.sqrt(np.r_[np.full(D, 0.5), np.full(2, 0.3)]))
            weights = np.linalg.lstsq(np.vstack([combined/np.sqrt(M), penalty]),
                                      np.r_[Yc[:, n]/np.sqrt(M), np.zeros(D+2)], rcond=None)[0]
            expected_population[:, n], expected_self[:, n] = weights[:D], weights[D:]
        np.testing.assert_allclose(model.input_factors@model.output_factors.T, expected_population, atol=3e-7)
        np.testing.assert_allclose(model.lag_weights, expected_self, atol=3e-7)

    def test_predict_and_rollout_match_dense_form_without_compressing_self_history(self):
        model = DiagonalLowRankVAR(2, 1).fit(self.train, start=4, max_iter=3)
        dense = VAR(2)
        dense.coefficients, dense.intercept = model.coefficients, model.intercept
        history = np.array([self.val[0][i:i+4] for i in range(3)])
        original = history.copy()
        np.testing.assert_allclose(model.rollout(history, 5), dense.rollout(history, 5), atol=1e-12)
        np.testing.assert_array_equal(history, original)
        self.assertEqual(model.parameter_count, 6+6+3+3)
        self.assertLessEqual(np.linalg.matrix_rank(model.input_factors@model.output_factors.T), 1)
        # The total VAR matrix is allowed to have full rank despite rank-one L.
        self.assertEqual(np.linalg.matrix_rank(model.coefficients.reshape(6, 3)), 3)

    def test_callback_stops_and_validation_selection_retains_best_iterate(self):
        calls = []
        model = DiagonalLowRankVAR(2, 1).fit(self.train, start=4, max_iter=10, tol=0,
            on_iteration=lambda m, i, obj: calls.append(i) or i < 2)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(model.n_iterations, 2)
        self.assertEqual(model.stopping_reason, 'callback')
        best, grid = select_diagonal_rank(self.train, self.val, np.full(3, -5.), P=2,
            ranks=[1, 2], population_lambdas=[0.1], context=4, max_iter=6, patience=2, verbose=False)
        for row in grid:
            expected = min(row['history'], key=lambda h: h['validation_mse_bounded'])
            self.assertEqual(row['selected_iteration'], expected['iteration'])
            self.assertEqual(row['validation']['bounded']['mse'][0], expected['validation_mse_bounded'])
            self.assertEqual(row['validation']['bounded']['n_windows'], 20)
        expected = min(grid, key=lambda row: row['validation']['bounded']['mse'][0])
        self.assertEqual(best.rank, expected['rank'])

    def test_saved_checkpoint_and_input_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root/'session'
            write_synthetic_session(directory, length=40)
            _, _, controls = run_controls(directory, context=4, lambdas=[0.01],
                                          output_dir=root/'controls', verbose=False)
            model, result, output = run(controls, dataset=directory, context=4, ranks=[1],
                population_lambdas=[0.1], max_iter=4, patience=2, output_dir=root/'first', verbose=False)
            self.assertEqual(json.loads((output/'selection.json').read_text()), result)
            self.assertNotIn('test', result)
            data = load_session(directory)
            restored, _ = load_checkpoint(output/'model.npz')
            expected = BoundedForecast(model, restored.floor)
            history = data.trials('val')[0][:4]
            np.testing.assert_array_equal(restored.rollout(history, 5), expected.rollout(history, 5))
            with self.assertRaises(FileExistsError):
                run(controls, dataset=directory, output_dir=output, verbose=False)
            with self.assertRaises(ValueError):
                run(controls, dataset=directory, context=5, ranks=[1], verbose=False)
            X = np.load(directory/'X_features.npy')
            X[17*40:] += 1000
            np.save(directory/'X_features.npy', X)
            with self.assertRaises(ValueError):
                run(controls, dataset=directory, context=4, ranks=[1], verbose=False)
            # With newly verified control provenance, changing only test rows
            # still leaves all fitted/selected parameters and validation equal.
            _, _, controls2 = run_controls(directory, context=4, lambdas=[0.01],
                                           output_dir=root/'controls2', verbose=False)
            other, second, _ = run(controls2, dataset=directory, context=4, ranks=[1],
                population_lambdas=[0.1], max_iter=4, patience=2, output_dir=root/'second', verbose=False)
            self.assertEqual(result['selected'], second['selected'])
            self.assertEqual(result['grid'], second['grid'])
            np.testing.assert_array_equal(model.coefficients, other.coefficients)

    def test_invalid_rank_settings_and_unfitted_use(self):
        for rank in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                DiagonalLowRankVAR(rank=rank)
        with self.assertRaises(RuntimeError):
            DiagonalLowRankVAR().predict(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            DiagonalLowRankVAR(rank=4).fit(self.train)
        with self.assertRaises(ValueError):
            DiagonalLowRankVAR(rank=1).fit(self.train, max_iter=-1)
        with self.assertRaises(ValueError):
            select_diagonal_rank(self.train, self.val, np.zeros(3), ranks=[], verbose=False)
        with self.assertRaises(ValueError):
            select_diagonal_rank(self.train, self.val, np.zeros(3), ranks=[1], patience=0, verbose=False)


if __name__ == '__main__':
    unittest.main()
