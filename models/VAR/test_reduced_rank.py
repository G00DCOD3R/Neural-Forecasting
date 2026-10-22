"""Numerical and data-separation checks for reduced-rank ridge VAR."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .data import load_session
from .data_example import write_synthetic_session
from .lag_selection import evaluate
from .reduced_rank import RankProjection, ReducedRankVAR
from .select_ranks import load_checkpoint, run, select_ranks
from .var_p import VAR


def design(trials, P, start):
    Z = np.array([np.concatenate([x[t-lag] for lag in range(1, P+1)])
                  for x in trials for t in range(start, len(x))])
    Y = np.array([x[t] for x in trials for t in range(start, len(x))])
    return Z, Y


class TestReducedRankVAR(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(49)
        mixing = np.array([[3., 1., 0.], [0., 0.4, 1.], [0., 0., 0.2]])
        self.train = [rng.normal(size=(80, 3))@mixing+2,
                      rng.normal(size=(65, 3))@mixing-1]
        self.val = [rng.normal(size=(35, 3))@mixing+0.5]

    def test_matches_independent_augmented_qr_svd_solution(self):
        # Independent algorithm: QR of the penalized design, then SVD in
        # orthonormal coordinates, rather than the implementation's eigenproblem.
        P, start, lam, rank = 2, 4, 0.17, 2
        Z, Y = design(self.train, P, start)
        M, D = Z.shape
        zmean, ymean = Z.mean(0), Y.mean(0)
        augmented_x = np.vstack([(Z-zmean)/np.sqrt(M), np.sqrt(lam)*np.eye(D)])
        augmented_y = np.vstack([(Y-ymean)/np.sqrt(M), np.zeros((D, Y.shape[1]))])
        Q, R = np.linalg.qr(augmented_x, mode='reduced')
        U, s, Vt = np.linalg.svd(Q.T@augmented_y, full_matrices=False)
        expected = np.linalg.solve(R, (U[:, :rank]*s[:rank])@Vt[:rank])
        model = ReducedRankVAR(P, rank, lam).fit(self.train, start=start)
        np.testing.assert_allclose(model.coefficients.reshape(D, 3), expected, atol=1e-11)
        np.testing.assert_allclose(model.intercept, ymean-zmean@expected, atol=1e-11)
        self.assertEqual(model.n_train_pairs, M)
        self.assertEqual(model.training_start, start)

    def test_full_rank_recovers_unrestricted_coefficients_and_forecasts(self):
        full = VAR(2, 0.1).fit(self.train, start=4)
        reduced = ReducedRankVAR(2, 3, 0.1).fit(self.train, start=4)
        np.testing.assert_allclose(reduced.coefficients, full.coefficients, atol=1e-12)
        np.testing.assert_allclose(reduced.intercept, full.intercept, atol=1e-12)
        np.testing.assert_allclose(reduced.rollout(self.val[0][:15], 5),
                                   full.rollout(self.val[0][:15], 5), atol=1e-12)

    def test_factor_predictions_and_recursive_forecasts_match_dense_form(self):
        reduced = ReducedRankVAR(2, 1, 0.1).fit(self.train, start=4)
        dense = VAR(2, 0.1)
        dense.coefficients, dense.intercept = reduced.coefficients, reduced.intercept
        history = np.array([self.val[0][i:i+4] for i in range(3)])
        original = history.copy()
        np.testing.assert_allclose(reduced.predict(history), dense.predict(history), atol=1e-12)
        np.testing.assert_allclose(reduced.rollout(history, 5), dense.rollout(history, 5), atol=1e-12)
        np.testing.assert_array_equal(history, original)
        self.assertEqual(reduced.parameter_count, (2*3+3)*1+3)
        self.assertEqual(reduced.coefficient_dimension, 1*(2*3+3-1)+3)
        self.assertLessEqual(np.linalg.matrix_rank(reduced.coefficients.reshape(6, 3)), 1)

    def test_penalized_objective_improves_with_rank_and_beats_naive_truncation(self):
        stats = VAR(2).statistics(self.train, start=4)
        before = [a.copy() for a in (stats.covariance, stats.cross_covariance,
                                    stats.past_mean, stats.target_mean)]
        projection = RankProjection.from_statistics(stats, 0.1)
        Z, Y = design(self.train, 2, 4)
        Z, Y = Z-Z.mean(0), Y-Y.mean(0)
        objective = lambda B: np.square(Y-Z@B).sum()/len(Y)+0.1*np.square(B).sum()
        scores = [objective(projection.model(r).coefficients.reshape(6, 3)) for r in (1, 2, 3)]
        self.assertTrue(np.all(np.diff(scores) <= 1e-12))
        full_weights = projection.full_model.coefficients.reshape(6, 3)
        U, s, Vt = np.linalg.svd(full_weights, full_matrices=False)
        naive = (U[:, :1]*s[:1])@Vt[:1]
        self.assertLess(scores[0], objective(naive)-1e-8)
        for actual, original in zip((stats.covariance, stats.cross_covariance,
                                     stats.past_mean, stats.target_mean), before):
            np.testing.assert_array_equal(actual, original)

    def test_selection_uses_common_targets_and_includes_exact_dense_reference(self):
        model, grid = select_ranks(iter(self.train), iter(self.val), P=2,
                                   ranks=[1, 2], lambdas=[0.01, 0.1], context=4, verbose=False)
        self.assertEqual(len(grid), 6)
        self.assertEqual({row['fitted_training_pairs'] for row in grid}, {137})
        self.assertEqual({row['validation']['n_windows'] for row in grid}, {31})
        for row in grid:
            if row['rank'] is None:
                expected = evaluate(VAR(2, row['lambda']).fit(self.train, start=4), self.val, context=4)
                self.assertEqual(row['validation'], expected)
        selected = min(grid, key=lambda row: row['validation']['mse'][0])
        self.assertEqual(model.lam, selected['lambda'])
        self.assertEqual(getattr(model, 'rank', None), selected['rank'])

    def test_saved_selection_round_trip_and_test_data_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root/'session'
            write_synthetic_session(directory, length=40)
            first, result, output = run(directory, P=2, ranks=[1, 2], lambdas=[0.01, 0.1],
                                        context=4, output_dir=root/'first', verbose=False)
            self.assertEqual(json.loads((output/'selection.json').read_text()), result)
            self.assertNotIn('test', result)
            self.assertEqual(result['phase'], 'validation_selection_only')
            data = load_session(directory)
            restored, preprocessing = load_checkpoint(output/'model.npz')
            np.testing.assert_array_equal(preprocessing['mu'], data.mu)
            np.testing.assert_array_equal(restored.predict(data.trials('val')[0][:4]),
                                          first.predict(data.trials('val')[0][:4]))
            with self.assertRaises(FileExistsError):
                run(directory, output_dir=output, verbose=False)
            X = np.load(directory/'X_features.npy')
            X[17*40:] += 1000
            np.save(directory/'X_features.npy', X)
            second, other, _ = run(directory, P=2, ranks=[1, 2], lambdas=[0.01, 0.1],
                                   context=4, output_dir=root/'second', verbose=False)
            self.assertEqual(result['selected'], other['selected'])
            self.assertEqual(result['grid'], other['grid'])
            np.testing.assert_array_equal(first.coefficients, second.coefficients)

    def test_loader_handles_both_families_and_rejects_bad_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.npz'
            for model in [VAR(2).fit(self.train, start=4), ReducedRankVAR(2, 1).fit(self.train, start=4)]:
                fields = dict(P=model.P, lam=model.lam, intercept=model.intercept,
                              n_train_pairs=model.n_train_pairs, training_start=4,
                              mu=np.zeros(3), sd=np.ones(3), log_transform=True)
                if isinstance(model, ReducedRankVAR):
                    fields.update(kind='reduced_rank_ridge', rank=model.rank,
                                  input_factors=model.input_factors, output_factors=model.output_factors)
                else:
                    fields.update(kind='unrestricted_ridge', coefficients=model.coefficients)
                np.savez_compressed(path, **fields)
                restored, _ = load_checkpoint(path)
                np.testing.assert_array_equal(restored.rollout(self.val[0][:4], 5),
                                              model.rollout(self.val[0][:4], 5))
                fields['sd'] = np.ones(2)
                np.savez_compressed(path, **fields)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)

    def test_invalid_ranks_and_unfitted_use(self):
        for rank in (0, -1, 1.2, True):
            with self.assertRaises(ValueError):
                ReducedRankVAR(rank=rank)
        with self.assertRaises(ValueError):
            ReducedRankVAR(rank=4).fit(self.train)
        with self.assertRaises(RuntimeError):
            ReducedRankVAR(rank=1).predict(self.val[0][:4])
        for ranks in ([], [0], [1.5], [True], [4]):
            with self.assertRaises(ValueError):
                select_ranks(self.train, self.val, ranks=ranks, verbose=False)
        projection = RankProjection.from_statistics(VAR(2).statistics(self.train), 0.1)
        for rank in (0, 4, True):
            with self.assertRaises(ValueError):
                projection.model(rank)


if __name__ == '__main__':
    unittest.main()
