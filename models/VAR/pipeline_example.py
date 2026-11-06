"""Verify the complete saved-run workflow on small synthetic session files."""
import argparse
from pathlib import Path
import tempfile

import numpy as np

try:
    from .data_example import write_synthetic_session
    from .train import run, load_checkpoint
except ImportError:
    from data_example import write_synthetic_session
    from train import run, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    print('SYNTHETIC PIPELINE ONLY: not a MICrONS experiment',flush=True)
    with tempfile.TemporaryDirectory(prefix='var_pipeline_example_') as temporary:
        directory = Path(temporary)
        write_synthetic_session(directory,seed=args.seed)
        model,results,output = run(directory,output_dir=args.output_dir,data_kind='synthetic')
        restored,preprocessing = load_checkpoint(output/'model.npz')
        raw_previous = np.load(directory/'X_features.npy')[-1].astype(np.float32)
        if preprocessing['log_transform']:
            raw_previous = np.log1p(np.clip(raw_previous,0,None))
        normalized = (raw_previous-preprocessing['mu'])/preprocessing['sd']
        np.testing.assert_array_equal(model.predict(normalized),restored.predict(normalized))
        print('Reload check: saved model produces identical one-step predictions.')
    print('Temporary synthetic inputs removed; saved run outputs remain.')
    print('These metrics use normalized activity and cannot be directly compared with the earlier raw synthetic MSE.')


if __name__ == '__main__':
    main()
