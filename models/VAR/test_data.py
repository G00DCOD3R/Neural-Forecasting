"""Processed-data checks and a miniature HDF5 timestamp-builder regression."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from .data import load_session


class TestSessionData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        rng = np.random.default_rng(42)
        self.X = (rng.normal(size=(200,3)) + 3).astype(np.float32)
        self.rows = pd.DataFrame({'trial_id': np.repeat(np.arange(20),10),
                                  'condition_hash': ['same_video']*200,
                                  'time': np.arange(200)/7.5})
        self.write()

    def write(self):
        np.save(self.directory/'X_features.npy',self.X)
        self.rows.to_csv(self.directory/'conditions.csv',index=False)

    def test_split_and_normalization_match_gvar_convention(self):
        data = load_session(self.directory)
        self.assertEqual(data.summary()['trials_per_split'], {'train':14,'val':3,'test':3})
        transformed = np.log1p(np.clip(self.X,0,None))
        mu,sd = transformed[:140].mean(0),transformed[:140].std(0)+1e-6
        np.testing.assert_array_equal(data.mu,mu)
        np.testing.assert_array_equal(data.sd,sd)
        np.testing.assert_array_equal(data.X,(transformed-mu)/sd)
        self.assertEqual(len(data.trials('train')),14)
        self.assertTrue(all(len(x)==10 for x in data.trials('train')))
        np.testing.assert_array_equal(data.trials('test')[0],data.X[170:180])

    def test_changing_holdout_values_does_not_change_training_preprocessing(self):
        before = load_session(self.directory)
        self.X[140:] += 1000
        self.write()
        after = load_session(self.directory)
        np.testing.assert_array_equal(before.mu,after.mu)
        np.testing.assert_array_equal(before.sd,after.sd)
        np.testing.assert_array_equal(before.X[:140],after.X[:140])

    def test_negative_values_are_counted_and_raw_mode_does_not_clip(self):
        self.X[0,0] = -2
        self.write()
        clipped = load_session(self.directory)
        raw_mode = load_session(self.directory,log_transform=False)
        self.assertEqual(clipped.negative_values,1)
        np.testing.assert_array_equal(raw_mode.mu,self.X[:140].mean(0))
        self.assertEqual(np.load(self.directory/'X_features.npy')[0,0],-2)

    def test_rejects_old_metadata_and_bad_ordering(self):
        self.rows.drop(columns='trial_id').to_csv(self.directory/'conditions.csv',index=False)
        with self.assertRaisesRegex(ValueError,'rebuild'):
            load_session(self.directory)
        self.rows.loc[10,'time'] = -1
        self.write()
        with self.assertRaisesRegex(ValueError,'chronologically'):
            load_session(self.directory)
        self.rows.loc[10,'time'] = 10/7.5
        self.rows.loc[10,'trial_id'] = 3
        self.write()
        with self.assertRaisesRegex(ValueError,'contiguous'):
            load_session(self.directory)

    def test_rejects_nonfinite_activity_and_wrong_row_count(self):
        self.X[0,0] = np.nan
        self.write()
        with self.assertRaisesRegex(ValueError,'NaN/Inf'):
            load_session(self.directory)
        self.X[0,0] = 1
        self.rows = self.rows.iloc[:-1]
        self.write()
        with self.assertRaisesRegex(ValueError,'one metadata row'):
            load_session(self.directory)


class TestTimestampBuilder(unittest.TestCase):
    def test_builder_orders_repeated_stimuli_by_time_and_preserves_neuron_columns(self):
        import build_dataset as builder
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame({'scan_session':[4.,4.,4.], 'nucleus_id':[101,102,103],
                          'unit_id':[11,12,13]}).to_pickle(directory/'node_data_v1.pkl')
            pd.DataFrame({'pre_nucleus_id':[101,102], 'post_nucleus_id':[102,103],
                          'n_synapses':[1,2]}).to_pickle(directory/'edge_data_v1.pkl')
            with h5py.File(directory/'microns.h5','w') as f:
                meta = f.create_group('sessions/4_7/meta')
                meta.create_dataset('unit_ids',data=[13,11,12])  # deliberately different order
                meta.create_dataset('condition_hashes',data=np.array(['A','B','A'],dtype='S'))
                for movie in ('A','B'):
                    video = f.create_group(f'videos/{movie}')
                    video.create_dataset('clip',data=np.zeros((20,2,2),dtype=np.uint8))
                    video.attrs['type'] = 'Clip'
                    video.create_group('instances')
                for index,movie in enumerate(('A','B','A')):
                    trial = f.create_group(f'sessions/4_7/trials/{index}')
                    response = np.repeat(np.array([[13],[11],[12]])+index*100,20,axis=1)
                    trial.create_dataset('responses',data=response)
                    trial.create_dataset('treadmill',data=np.zeros(20))
                    trial.create_dataset('pupil',data=np.zeros((4,20)))
                    trial.create_dataset('stim_times',data=index*5+np.arange(20)/7.5)
                    f[f'videos/{movie}/instances'][f'4_7_tr{index}'] = h5py.SoftLink(trial.name)
            # Processing A's two trials first would yield chronological order 0,2,1
            # without the timestamp correction, so make that order deterministic.
            with patch.object(builder.MicronsReader,'get_hashes_by_session',return_value=['A','B']):
                with redirect_stdout(io.StringIO()):
                    builder.build_dataset(4.,'4_7',data_dir=str(directory),
                                          out_dir=str(directory/'processed'),out_name='fixture')
            output = directory/'processed/fixture'
            rows = pd.read_csv(output/'conditions.csv')
            self.assertTrue(rows['time'].is_monotonic_increasing)
            np.testing.assert_array_equal(rows['trial_id'],np.repeat([0,1,2],20))
            np.testing.assert_array_equal(rows['condition_hash'],np.repeat(['A','B','A'],20))
            X = np.load(output/'X_features.npy')
            np.testing.assert_array_equal(X[[0,20,40]],[[11,12,13],[111,112,113],[211,212,213]])


if __name__ == '__main__':
    unittest.main()
