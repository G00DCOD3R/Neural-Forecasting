"""End-to-end run, artifact replay and held-out-data separation checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .data import load_session
from .data_example import write_synthetic_session
from .evaluation import evaluate_recursive
from .train import run,load_checkpoint


class TestTrainingRun(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root/'session'
        write_synthetic_session(self.dataset,length=40)

    def test_saved_model_reproduces_metrics_and_normalization(self):
        output = self.root/'run'
        model,results,_ = run(self.dataset,output,lambdas=[0.01,0.1],data_kind='synthetic',verbose=False)
        restored,preprocessing = load_checkpoint(output/'model.npz')
        data = load_session(self.dataset)
        np.testing.assert_array_equal(preprocessing['mu'],data.mu)
        np.testing.assert_array_equal(preprocessing['sd'],data.sd)
        repeated = evaluate_recursive(restored,data.trials('test'),mean_prediction=np.zeros(3))
        self.assertEqual(repeated,results['test'])
        np.testing.assert_array_equal(restored.coefficients,model.coefficients)
        self.assertEqual(results['n_train_pairs'],14*39)
        self.assertEqual(results['test']['n_windows'],3*(40-15-5+1))
        chosen = min(results['grid'],key=lambda row:row['val_mse'])
        self.assertEqual(results['lam'],chosen['lam'])
        self.assertEqual(json.loads((output/'results.json').read_text())['config']['data_kind'],'synthetic')
        self.assertTrue((output/'metrics_summary.txt').is_file())
        self.assertEqual(len(results['data']['sha256']['X_features.npy']),64)

    def test_changing_test_data_does_not_change_selected_model(self):
        first,_,out1 = run(self.dataset,self.root/'first',lambdas=[0.01,0.1],verbose=False)
        X = np.load(self.dataset/'X_features.npy')
        X[17*40:] += 100
        np.save(self.dataset/'X_features.npy',X)
        second,_,out2 = run(self.dataset,self.root/'second',lambdas=[0.01,0.1],verbose=False)
        self.assertEqual(first.lam,second.lam)
        np.testing.assert_array_equal(first.coefficients,second.coefficients)
        np.testing.assert_array_equal(first.intercept,second.intercept)
        _,norm1 = load_checkpoint(out1/'model.npz')
        _,norm2 = load_checkpoint(out2/'model.npz')
        np.testing.assert_array_equal(norm1['mu'],norm2['mu'])
        np.testing.assert_array_equal(norm1['sd'],norm2['sd'])

    def test_rejects_nonempty_output_and_missing_data(self):
        output = self.root/'existing'
        output.mkdir()
        (output/'keep.txt').write_text('preserve this run')
        with self.assertRaises(FileExistsError):
            run(self.dataset,output,verbose=False)
        self.assertEqual((output/'keep.txt').read_text(),'preserve this run')
        with self.assertRaises(FileNotFoundError):
            run(self.root/'missing',self.root/'not_created',verbose=False)
        self.assertFalse((self.root/'not_created').exists())

    def test_mean_baseline_matches_zero_in_normalized_space(self):
        _,results,_ = run(self.dataset,self.root/'mean',lambdas=[0.1],verbose=False)
        data = load_session(self.dataset)
        from .evaluation import forecast_batches
        target = np.concatenate([y for _,y in forecast_batches(data.trials('test'))])
        np.testing.assert_allclose(results['test']['mean']['mse'],(target**2).mean((0,2)))
        np.testing.assert_allclose(results['test']['mean']['pearson'],np.zeros(5))


if __name__ == '__main__':
    unittest.main()
