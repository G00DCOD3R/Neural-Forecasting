"""Run with Python's built-in unittest; no extra testing packages required."""
import unittest

import numpy as np

from .synthetic_example import generate_trials
from .var import VAR1
from .selection import evaluate_one_step, select_ridge
from .evaluation import forecast_batches, evaluate_recursive


class TestVAR1(unittest.TestCase):
    def test_fit_matches_independent_augmented_least_squares(self):
        rng = np.random.default_rng(8)
        trials = [rng.normal(size=(35, 3)) + 2, rng.normal(size=(21, 3)) - 1]
        Z = np.vstack([x[:-1] for x in trials])
        Y = np.vstack([x[1:] for x in trials])
        M, N = Z.shape
        lam = 0.13
        # Independent reference: express ridge as additional observations and
        # solve with least squares, leaving the intercept's penalty at zero.
        design = np.column_stack([Z, np.ones(M)]) / np.sqrt(M)
        penalty = np.column_stack([np.sqrt(lam)*np.eye(N), np.zeros(N)])
        expected = np.linalg.lstsq(np.vstack([design, penalty]),
                                  np.vstack([Y/np.sqrt(M), np.zeros((N,N))]), rcond=None)[0]
        model = VAR1(lam).fit(trials)
        np.testing.assert_allclose(model.coefficients, expected[:-1], atol=1e-12)
        np.testing.assert_allclose(model.intercept, expected[-1], atol=1e-12)
        np.testing.assert_allclose(model.predict(Z), Z @ expected[:-1] + expected[-1], atol=1e-12)

    def test_recovers_directed_population_interactions(self):
        trials, coefficients, intercept = generate_trials(seed=42)
        model = VAR1(lam=1e-5).fit(trials[:14])
        np.testing.assert_allclose(model.coefficients, coefficients, atol=0.04)
        np.testing.assert_allclose(model.intercept, intercept, atol=0.15)
        self.assertGreater(model.coefficients[0,1], 0.5)  # A drives B
        self.assertLess(abs(model.coefficients[1,0]), 0.04)  # not B driving A
        self.assertEqual(model.parameter_count, 12)

    def test_training_does_not_join_trial_boundaries(self):
        trials = [np.array([[0.],[1.]]), np.array([[100.],[101.]]), np.array([[-999.]])]
        model = VAR1(lam=1e-6).fit(trials)
        self.assertEqual(model.n_train_pairs, 2)
        np.testing.assert_allclose(model.predict([10.]), [11.], atol=1e-6)

    def test_larger_ridge_penalty_shrinks_coefficients(self):
        trials, _, _ = generate_trials(seed=1, count=2, length=100)
        weak = VAR1(lam=1e-5).fit(trials)
        strong = VAR1(lam=1).fit(trials)
        self.assertLess(np.linalg.norm(strong.coefficients), np.linalg.norm(weak.coefficients))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            VAR1(lam=0)
        with self.assertRaises(RuntimeError):
            VAR1().predict([1.])
        for trials in ([], [np.zeros((1,3))], [np.array([[np.nan],[1.]])],
                       [np.zeros((3,2)), np.zeros((3,4))]):
            with self.subTest(trials=trials):
                with self.assertRaises(ValueError):
                    VAR1().fit(trials)
        model = VAR1().fit([np.zeros((3,2))])
        with self.assertRaises(ValueError):
            model.predict([1.,2.,3.])


class TestRidgeSelection(unittest.TestCase):
    def test_selection_uses_validation_but_fits_training_only(self):
        trials, _, _ = generate_trials(seed=10, count=6, length=100)
        train, validation = trials[:4], trials[4:]
        penalties = [1e-5, 0.01, 1.0]
        expected = [evaluate_one_step(VAR1(lam).fit(train), validation)['mse'] for lam in penalties]
        model, scores = select_ridge(iter(train), iter(validation), iter(penalties))
        self.assertEqual(model.lam, penalties[int(np.argmin(expected))])
        np.testing.assert_allclose([row['val_mse'] for row in scores], expected)
        reference = VAR1(model.lam).fit(train)
        np.testing.assert_array_equal(model.coefficients, reference.coefficients)
        np.testing.assert_array_equal(model.intercept, reference.intercept)
        self.assertEqual(model.n_train_pairs, 4*99)

    def test_evaluation_weights_observations_and_keeps_trials_separate(self):
        model = VAR1().fit([np.zeros((3,1))])  # predicts zero
        trials = [np.array([[0.],[2.],[4.]]), np.array([[100.],[3.]])]
        result = evaluate_one_step(model, trials)
        self.assertEqual(result['n_pairs'], 3)
        self.assertAlmostEqual(result['mse'], (4+16+9)/3)
        self.assertAlmostEqual(result['copy_last_mse'], (4+4+97**2)/3)
        # No cross-trial pair is scored; context=2 leaves only target 4.
        result = evaluate_one_step(model, trials, context=2)
        self.assertEqual(result['n_pairs'], 1)
        self.assertEqual(result['mse'], 16)

    def test_rejects_empty_validation_and_invalid_penalties(self):
        train = [np.zeros((3,1))]
        for penalties in ([], [0], [np.nan], [-1]):
            with self.subTest(penalties=penalties):
                with self.assertRaises(ValueError):
                    select_ridge(train, train, penalties)
        with self.assertRaises(ValueError):
            select_ridge(train, [], [0.1])


class TestRecursiveForecast(unittest.TestCase):
    @staticmethod
    def scalar_model():
        model = VAR1()
        model.coefficients = np.array([[0.5]])
        model.intercept = np.array([1.0])
        return model

    def test_rollout_matches_closed_form_and_preserves_input(self):
        model = self.scalar_model()
        previous = np.array([[4.],[8.]])
        original = previous.copy()
        np.testing.assert_allclose(model.rollout(previous, 3),
                                   [[[3.],[2.5],[2.25]], [[5.],[3.5],[2.75]]])
        np.testing.assert_array_equal(previous, original)
        self.assertEqual(model.rollout(previous[0], 3).shape, (3,1))

    def test_windows_match_explicit_trial_slices(self):
        trials = [np.arange(14).reshape(7,2), 100+np.arange(12).reshape(6,2)]
        batches = list(forecast_batches(trials, context=3, horizon=3, batch_size=1))
        previous = np.concatenate([x for x,_ in batches])
        target = np.concatenate([y for _,y in batches])
        expected = [(x[t-1],x[t:t+3]) for x in trials for t in range(3,len(x)-3+1)]
        np.testing.assert_array_equal(previous, np.stack([x for x,_ in expected]))
        np.testing.assert_array_equal(target, np.stack([y for _,y in expected]))
        self.assertEqual(len(target), 3)

    def test_future_observations_do_not_affect_fixed_origin_forecast(self):
        model = self.scalar_model()
        original = np.arange(7.)[:,None]
        altered = original.copy()
        altered[3:] += 1000
        p1,y1 = next(forecast_batches([original],context=3,horizon=3,batch_size=1))
        p2,y2 = next(forecast_batches([altered],context=3,horizon=3,batch_size=1))
        np.testing.assert_array_equal(model.rollout(p1,3),model.rollout(p2,3))
        self.assertFalse(np.array_equal(y1,y2))

    def test_streaming_metrics_match_direct_calculation(self):
        rng = np.random.default_rng(11)
        trials = [rng.normal(size=(24,3)),rng.normal(size=(19,3))]
        for x in trials:
            x[:,2] = 0  # constant target and prediction test Pearson convention
        model = VAR1()
        model.coefficients = np.diag([0.5,0.3,0.0])
        model.intercept = np.zeros(3)
        result = evaluate_recursive(model,trials,context=3,horizon=4,batch_size=3)
        batches = list(forecast_batches(trials,context=3,horizon=4,batch_size=100))
        previous = np.concatenate([x for x,_ in batches])
        target = np.concatenate([y for _,y in batches])
        for name,pred in [('var',model.rollout(previous,4)),
                          ('copy_last',np.broadcast_to(previous[:,None],target.shape))]:
            err = pred-target
            pc,tc = pred-pred.mean(0),target-target.mean(0)
            corr = (pc*tc).sum(0)/(np.linalg.norm(pc,axis=0)*np.linalg.norm(tc,axis=0)+1e-8)
            np.testing.assert_allclose(result[name]['mse'],(err**2).mean((0,2)),atol=1e-12)
            np.testing.assert_allclose(result[name]['mae'],abs(err).mean((0,2)),atol=1e-12)
            np.testing.assert_allclose(result[name]['pearson'],corr.mean(1),atol=1e-12)
            np.testing.assert_allclose(result[name]['mse_per_neuron_h1'],(err[:,0]**2).mean(0),atol=1e-12)
        self.assertEqual(result['n_windows'],31)

    def test_horizon_one_agrees_with_existing_evaluation(self):
        model = self.scalar_model()
        trials = [np.arange(9.)[:,None],np.arange(12.)[:,None]]
        reference = evaluate_one_step(model,trials,context=3)
        result = evaluate_recursive(model,trials,context=3,horizon=1,batch_size=2)
        self.assertAlmostEqual(result['var']['mse'][0],reference['mse'])
        self.assertAlmostEqual(result['copy_last']['mse'][0],reference['copy_last_mse'])
        self.assertEqual(result['n_windows'],reference['n_pairs'])

    def test_rejects_invalid_horizon_and_empty_windows(self):
        model = self.scalar_model()
        for horizon in (0,-1,1.5):
            with self.subTest(horizon=horizon):
                with self.assertRaises(ValueError):
                    model.rollout([1.],horizon)
        with self.assertRaises(ValueError):
            evaluate_recursive(model,[np.zeros((10,1))],context=15,horizon=5)


if __name__ == '__main__':
    unittest.main()
