"""Independent fitting, bound-feedback and validation-protocol checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .compare_controls import evaluate_policies, load_checkpoint, run, select_controls
from .data import load_session
from .data_example import write_synthetic_session
from .independent_ar import IndependentAR
from .lag_selection import evaluate
from .prediction_bounds import BoundedForecast, activity_floor
from .var_p import VAR


class TestVARControls(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(38)
        self.train = [rng.normal(size=(35, 3))+1, rng.normal(size=(29, 3))-2]
        self.val = [rng.normal(size=(22, 3))+0.5]

    def test_independent_ar_matches_separate_augmented_least_squares(self):
        P, start, lam = 2, 4, 0.1
        stats = VAR(P).statistics(self.train, start)
        covariance = stats.covariance.copy()
        cross = stats.cross_covariance.copy()
        model = IndependentAR(P, lam).fit_statistics(stats)
        for n in range(3):
            Z = np.array([[x[t-1, n], x[t-2, n]] for x in self.train for t in range(start, len(x))])
            y = np.array([x[t, n] for x in self.train for t in range(start, len(x))])
            M = len(y)
            augmented = np.vstack([np.column_stack([Z, np.ones(M)])/np.sqrt(M),
                                   np.column_stack([np.sqrt(lam)*np.eye(P), np.zeros(P)])])
            expected = np.linalg.lstsq(augmented, np.r_[y/np.sqrt(M), np.zeros(P)], rcond=None)[0]
            np.testing.assert_allclose(model.lag_weights[:, n], expected[:-1], atol=1e-12)
            np.testing.assert_allclose(model.intercept[n], expected[-1], atol=1e-12)
        np.testing.assert_array_equal(stats.covariance, covariance)
        np.testing.assert_array_equal(stats.cross_covariance, cross)
        self.assertEqual(model.parameter_count, 9)
        self.assertEqual(model.n_train_pairs, 56)

    def test_predictions_ignore_other_neurons_and_match_dense_diagonal_form(self):
        independent = IndependentAR(2, 0.1).fit(self.train, start=4)
        dense = VAR(2, 0.1)
        dense.coefficients = np.zeros((2, 3, 3))
        dense.coefficients[:, np.arange(3), np.arange(3)] = independent.lag_weights
        dense.intercept = independent.intercept
        history = np.array([self.val[0][i:i+4] for i in range(3)])
        np.testing.assert_allclose(independent.rollout(history, 5), dense.rollout(history, 5), atol=1e-12)
        changed = history.copy()
        changed[:, :, 1:] += 100
        np.testing.assert_array_equal(independent.predict(changed)[:, 0], independent.predict(history)[:, 0])
        changed_train = [x.copy() for x in self.train]
        for x in changed_train:
            x[:, 1:] *= 10
        refitted = IndependentAR(2, 0.1).fit(changed_train, start=4)
        np.testing.assert_allclose(refitted.lag_weights[:, 0], independent.lag_weights[:, 0], atol=1e-12)

    def test_bound_is_transformed_raw_zero_and_preserves_valid_negative_z_scores(self):
        floor = activity_floor(np.array([2.], np.float32), np.array([1.], np.float32))
        np.testing.assert_array_equal(floor, [-2.])
        base = VAR(1)
        base.coefficients, base.intercept = np.zeros((1, 1, 1)), np.array([-1.])
        np.testing.assert_array_equal(BoundedForecast(base, floor).predict([[0.]]), [-1.])
        base.intercept = np.array([-3.])
        np.testing.assert_array_equal(BoundedForecast(base, floor).predict([[0.]]), [-2.])

    def test_recursive_bound_is_applied_before_feedback(self):
        base = VAR(1)
        base.coefficients, base.intercept = np.array([[[-1.]]]), np.array([-1.])
        history = np.array([[2.]])
        original = history.copy()
        bounded = BoundedForecast(base, [0.])
        np.testing.assert_array_equal(bounded.rollout(history, 3), [[0.], [0.], [0.]])
        np.testing.assert_array_equal(np.maximum(base.rollout(history, 3), 0), [[0.], [2.], [0.]])
        np.testing.assert_array_equal(bounded.rollout(history[None], 3), [[[0.], [0.], [0.]]])
        np.testing.assert_array_equal(history, original)

    def test_paired_metrics_use_common_origins_and_match_existing_evaluator(self):
        model = VAR(2, 0.1).fit(self.train, start=4)
        floor = np.array([-1., -2., -3.])
        both = evaluate_policies(model, self.val, floor, context=4, batch_size=3)
        self.assertEqual(both['unbounded'], evaluate(model, self.val, context=4, batch_size=3))
        self.assertEqual(both['bounded'], evaluate(BoundedForecast(model, floor), self.val, context=4, batch_size=3))
        self.assertEqual(both['bounded']['n_windows'], 18)

    def test_selection_keeps_common_targets_and_selects_each_family_separately(self):
        models, grid = select_controls(iter(self.train), iter(self.val), np.full(3, -5.),
                                       P=2, lambdas=[0.01, 0.1], context=4, verbose=False)
        self.assertEqual(len(grid), 4)
        self.assertEqual({row['fitted_training_pairs'] for row in grid}, {56})
        self.assertEqual({row['validation']['bounded']['n_windows'] for row in grid}, {18})
        for family, model in models.items():
            candidates = [row for row in grid if row['family'] == family]
            expected = min(candidates, key=lambda row: row['validation']['bounded']['mse'][0])
            self.assertEqual(model.lam, expected['lambda'])
        reference = VAR(2, models['dense_var'].lam).fit(self.train, start=4)
        np.testing.assert_array_equal(reference.coefficients, models['dense_var'].coefficients)

    def test_both_checkpoints_replay_and_test_data_cannot_change_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root/'session'
            write_synthetic_session(directory, length=40)
            first, results, output = run(directory, lambdas=[0.01, 0.1], context=4,
                                         output_dir=root/'first', verbose=False)
            self.assertNotIn('test', results)
            self.assertEqual(json.loads((output/'selection.json').read_text()), results)
            data = load_session(directory)
            floor = activity_floor(data.mu, data.sd)
            for family, base in first.items():
                restored, prep, restored_family = load_checkpoint(output/f'{family}.npz')
                self.assertEqual(family, restored_family)
                np.testing.assert_array_equal(prep['mu'], data.mu)
                expected = BoundedForecast(base, floor)
                history = data.trials('val')[0][:4]
                np.testing.assert_array_equal(restored.rollout(history, 5), expected.rollout(history, 5))
            for row in results['grid']:
                for split in ('train', 'validation'):
                    self.assertLessEqual(row[split]['bounded']['mse'][0], row[split]['unbounded']['mse'][0]+1e-12)
            with self.assertRaises(FileExistsError):
                run(directory, output_dir=output, verbose=False)
            X = np.load(directory/'X_features.npy')
            X[17*40:] += 1000
            np.save(directory/'X_features.npy', X)
            second, other, _ = run(directory, lambdas=[0.01, 0.1], context=4,
                                    output_dir=root/'second', verbose=False)
            self.assertEqual(results['grid'], other['grid'])
            self.assertEqual(results['selected'], other['selected'])
            np.testing.assert_array_equal(first['dense_var'].coefficients, second['dense_var'].coefficients)

    def test_invalid_bounds_settings_and_unfitted_models(self):
        with self.assertRaises(RuntimeError):
            IndependentAR().predict(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            IndependentAR(2).fit_statistics(VAR(1).statistics(self.train))
        for mu, sd, log in [([1.], [1.], False), ([1.], [0.], True), ([1.], [np.nan], True),
                            ([1., 2.], [1.], True)]:
            with self.assertRaises(ValueError):
                activity_floor(mu, sd, log)
        base = VAR(2).fit(self.train)
        with self.assertRaises(ValueError):
            BoundedForecast(base, [-1.])
        with self.assertRaises(ValueError):
            select_controls(self.train, self.val, np.zeros(3), lambdas=[], verbose=False)
        with self.assertRaises(ValueError):
            select_controls(self.train, self.val, np.zeros(3), context=1, verbose=False)


if __name__ == '__main__':
    unittest.main()
