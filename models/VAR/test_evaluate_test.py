"""Frozen test evaluation: no fitting, common targets and provenance guards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import evaluate_test
from .compare_controls import run as make_controls
from .data_example import write_synthetic_session
from .diagonal_low_rank import DiagonalLowRankVAR
from .independent_ar import IndependentAR
from .select_diagonal_rank import run as make_selected
from .train import sha256
from .var_p import VAR


class TestFrozenEvaluation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root/'session'
        write_synthetic_session(self.data, length=40)
        _, _, self.controls = make_controls(self.data, context=4, lambdas=[0.01],
            output_dir=self.root/'controls', verbose=False)
        _, _, self.selected = make_selected(self.controls, dataset=self.data, context=4,
            ranks=[1], population_lambdas=[0.1], max_iter=3, patience=2,
            output_dir=self.root/'selected', verbose=False)

    def test_no_refitting_and_ledger_precedes_test_forecasts(self):
        output = self.root/'test'
        checkpoint_paths = [self.selected/'model.npz', self.controls/'dense_var.npz',
                            self.controls/'independent_ar.npz']
        before = {str(p): sha256(p) for p in checkpoint_paths}
        original_score = evaluate_test.score_frozen

        def guarded_score(*args, **kwargs):
            protocol = json.loads((output/'frozen_protocol.json').read_text())
            self.assertFalse(protocol['refit'])
            self.assertFalse(protocol['test_based_selection'])
            self.assertEqual(protocol['primary_prediction_policy'], 'bounded')
            self.assertEqual(protocol['selected_configuration']['rank'], 1)
            self.assertEqual(protocol['horizon'], 1)
            return original_score(*args, **kwargs)

        with patch.object(VAR, 'fit', side_effect=AssertionError('Refitting prohibited')), \
             patch.object(VAR, 'statistics', side_effect=AssertionError('Training statistics prohibited')), \
             patch.object(IndependentAR, 'fit_statistics', side_effect=AssertionError('Refitting prohibited')), \
             patch.object(DiagonalLowRankVAR, 'fit_statistics', side_effect=AssertionError('Refitting prohibited')), \
             patch.object(evaluate_test, 'score_frozen', side_effect=guarded_score) as score:
            result, saved = evaluate_test.run(self.selected, dataset=self.data,
                                              output_dir=output, verbose=False)
        self.assertEqual(score.call_count, 1)
        self.assertEqual(result['phase'], 'frozen_test_evaluation')
        self.assertEqual(result, json.loads((saved/'results.json').read_text()))
        self.assertEqual(result['protocol']['expected_test_windows'], 108)
        self.assertEqual(result['protocol']['test_trial_ids'], [17, 18, 19])
        for row in result['models'].values():
            self.assertEqual(row['test']['bounded']['n_windows'], 108)
        self.assertEqual(before, {str(p): sha256(p) for p in checkpoint_paths})

    def test_metric_values_match_independent_direct_arrays(self):
        trials = [np.arange(14.).reshape(7, 2)/10, np.arange(16.).reshape(8, 2)/10+2]
        model = IndependentAR(2)
        model.lag_weights, model.intercept = np.array([[0.5, 0.2], [0.1, -0.3]]), np.array([-0.2, 0.1])
        floor = np.array([0., -1.])
        metrics, baselines = evaluate_test.score_frozen({'control': model}, trials, floor, context=4, batch_size=2)
        target = np.array([x[t] for x in trials for t in range(4, len(x))])[:, None]
        history = np.array([x[t-2:t] for x in trials for t in range(4, len(x))])
        raw = np.array([0.5*h[-1, 0]+0.1*h[-2, 0]-0.2 for h in history])
        raw2 = np.array([0.2*h[-1, 1]-0.3*h[-2, 1]+0.1 for h in history])
        prediction = np.maximum(np.column_stack([raw, raw2])[:, None], floor)
        error = prediction-target
        pc, tc = prediction-prediction.mean(0), target-target.mean(0)
        m = metrics['control']['bounded']
        np.testing.assert_allclose(m['mse'], np.square(error).mean((0, 2)), atol=1e-12)
        np.testing.assert_allclose(m['mae'], np.abs(error).mean((0, 2)), atol=1e-12)
        pearson = ((pc*tc).sum(0)/(np.linalg.norm(pc, axis=0)*np.linalg.norm(tc, axis=0)+1e-8)).mean(-1)
        np.testing.assert_allclose(m['pearson'], pearson, atol=1e-12)
        np.testing.assert_allclose(baselines['copy_last']['mse'], np.square(history[:, -1:, :]-target).mean((0, 2)), atol=1e-12)
        np.testing.assert_allclose(baselines['training_mean']['mse'], np.square(target).mean((0, 2)), atol=1e-12)
        self.assertEqual(m['n_windows'], 7)

    def test_parameter_mismatch_is_rejected_before_test_scoring(self):
        path = self.selected/'selection.json'
        selected = json.loads(path.read_text())
        selected['selected']['rank'] = 2
        path.write_text(json.dumps(selected))
        with patch.object(evaluate_test, 'score_frozen') as score:
            with self.assertRaises(ValueError):
                evaluate_test.run(self.selected, dataset=self.data, output_dir=self.root/'bad', verbose=False)
            score.assert_not_called()

    def test_changed_source_or_inputs_are_rejected(self):
        path = self.selected/'selection.json'
        selected = json.loads(path.read_text())
        wrong = json.loads(path.read_text())
        wrong['source_sha256']['diagonal_low_rank.py'] = '0'*64
        path.write_text(json.dumps(wrong))
        with self.assertRaises(ValueError):
            evaluate_test.run(self.selected, dataset=self.data, output_dir=self.root/'bad_source', verbose=False)
        path.write_text(json.dumps(selected))
        X = np.load(self.data/'X_features.npy')
        X[17*40:] += 1000
        np.save(self.data/'X_features.npy', X)
        with self.assertRaises(ValueError):
            evaluate_test.run(self.selected, dataset=self.data, output_dir=self.root/'bad_data', verbose=False)

    def test_control_weights_cannot_change_after_selection(self):
        path = self.controls/'independent_ar.npz'
        with np.load(path, allow_pickle=False) as saved:
            fields = {name: saved[name].copy() for name in saved.files}
        fields['lag_weights'] += 0.1
        np.savez_compressed(path, **fields)
        with patch.object(evaluate_test, 'score_frozen') as score:
            with self.assertRaises(ValueError):
                evaluate_test.run(self.selected, dataset=self.data, output_dir=self.root/'bad_weights', verbose=False)
            score.assert_not_called()

    def test_output_is_not_overwritten_and_batch_size_cannot_change_scores(self):
        first, output = evaluate_test.run(self.selected, dataset=self.data, batch_size=7,
                                          output_dir=self.root/'first', verbose=False)
        with self.assertRaises(FileExistsError):
            evaluate_test.run(self.selected, dataset=self.data, output_dir=output, verbose=False)
        second, _ = evaluate_test.run(self.selected, dataset=self.data, batch_size=19,
                                      output_dir=self.root/'second', verbose=False)
        for name in first['models']:
            for policy in ('unbounded', 'bounded'):
                for metric in ('mse', 'mae', 'pearson', 'r2'):
                    np.testing.assert_allclose(first['models'][name]['test'][policy][metric],
                                               second['models'][name]['test'][policy][metric], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
