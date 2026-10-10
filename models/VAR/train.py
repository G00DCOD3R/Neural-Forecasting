"""Load a processed session, select a ridge VAR, evaluate and save the run."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
import scipy

try:
    from .data import load_session
    from .evaluation import evaluate_recursive
    from .selection import select_ridge
    from .var import VAR1
except ImportError:
    from data import load_session
    from evaluation import evaluate_recursive
    from selection import select_ridge
    from var import VAR1

RUNS_DIR = Path(__file__).resolve().parent / 'runs'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    """Restore VAR and its training-fitted preprocessing from model.npz."""
    with np.load(path, allow_pickle=False) as saved:
        if int(saved['P']) != 1:
            raise ValueError('This implementation supports one lag only')
        model = VAR1(lam=float(saved['lam']))
        model.coefficients = saved['coefficients'].copy()
        model.intercept = saved['intercept'].copy()
        model.n_train_pairs = int(saved['n_train_pairs'])
        preprocessing = {'mu': saved['mu'].copy(), 'sd': saved['sd'].copy(),
                         'log_transform': bool(saved['log_transform'])}
    N = len(model.intercept)
    if (model.coefficients.shape != (N,N) or preprocessing['mu'].shape != (N,)
            or preprocessing['sd'].shape != (N,)
            or not np.isfinite(model.coefficients).all() or not np.isfinite(model.intercept).all()
            or not np.isfinite(preprocessing['mu']).all() or not np.isfinite(preprocessing['sd']).all()
            or np.any(preprocessing['sd'] <= 0)):
        raise ValueError('Checkpoint has invalid model or preprocessing dimensions/values')
    return model, preprocessing


def summary_text(results):
    cfg = results['config']
    lines = [f'Data kind: {cfg["data_kind"]}; lag order: 1; selected lambda: {results["lam"]:g}',
             f'Metric space: {results["metric_space"]}',
             f'Activity shape: {results["data"]["shape"]}; parameters: {results["parameter_count"]:,}',
             f'Training pairs: {results["n_train_pairs"]}; test forecast windows: {results["test"]["n_windows"]}',
             '', 'lambda        validation MSE']
    lines.extend(f'{row["lam"]:<13g} {row["val_mse"]:.9f}' for row in results['grid'])
    lines.extend(['', f'{"h":>3} {"VAR MSE":>12} {"copy MSE":>12} {"mean MSE":>12} {"VAR Pearson":>12}'])
    for h in range(cfg['horizon']):
        metrics = results['test']
        lines.append(f'{h+1:3d} {metrics["var"]["mse"][h]:12.6f}'
                     f' {metrics["copy_last"]["mse"][h]:12.6f}'
                     f' {metrics["mean"]["mse"][h]:12.6f}'
                     f' {metrics["var"]["pearson"][h]:12.4f}')
    return '\n'.join(lines) + '\n'


def run(dataset='cave_all', output_dir=None, lambdas=(0.01,0.1,1.0), context=15,
        horizon=5, batch_size=256, log_transform=True, val_frac=0.15, test_frac=0.15,
        data_kind='unspecified', verbose=True):
    """Fit training trials, select with validation, test once and save artifacts."""
    if data_kind not in ('synthetic','microns','unspecified'):
        raise ValueError('data_kind must be synthetic, microns or unspecified')
    if any(not isinstance(v,(int,np.integer)) or v < 1 for v in (context,horizon,batch_size)):
        raise ValueError('context, horizon and batch_size must be positive integers')
    lambdas = [float(lam) for lam in lambdas]
    if not lambdas or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError('Provide at least one finite positive ridge penalty')
    output = Path(output_dir or RUNS_DIR/f'run_{uuid4().hex[:8]}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'{output} is not an empty directory; choose another output path')
    started = time.perf_counter()
    data = load_session(dataset,log_transform,val_frac,test_frac)
    if verbose:
        print(f'Loaded {tuple(data.X.shape)} activity ({data_kind}); '
              f'trials: {data.summary()["trials_per_split"]}', flush=True)
        print('Selecting ridge penalty by one-step validation MSE...', flush=True)
    model, grid = select_ridge(data.trials('train'),data.trials('val'),lambdas,context=context)
    if verbose:
        print(f'Selected lambda={model.lam:g}; evaluating held-out {horizon}-step forecasts...', flush=True)
    # In training-standardized coordinates, zero represents the fitted training
    # mean, as in GVAR. Add this baseline to both VAR and copy-last.
    test = evaluate_recursive(model,data.trials('test'),context,horizon,batch_size,
                              mean_prediction=np.zeros(data.X.shape[1]))
    metric_space = 'training-standardized log1p(clipped activity)' if log_transform else 'training-standardized activity'
    source_files = ['var.py','data.py','selection.py','evaluation.py','train.py']
    results = {
        'config': {'dataset':str(dataset), 'data_kind':data_kind, 'P':1, 'lambdas':lambdas,
                   'context':context, 'horizon':horizon, 'batch_size':batch_size,
                   'log_transform':bool(log_transform), 'val_frac':val_frac, 'test_frac':test_frac},
        'P':1, 'lam':model.lam, 'parameter_count':model.parameter_count,
        'n_train_pairs':model.n_train_pairs, 'grid':grid, 'test':test,
        'data': {**data.summary(),
                 'sha256':{name:sha256(data.directory/name) for name in ('X_features.npy','conditions.csv')},
                 'split_trial_ids':{s:np.flatnonzero(data.split==s).tolist() for s in ('train','val','test')}},
        'metric_space':metric_space,
        'pearson_convention':'mean across all neurons, including zero for constant pairs, epsilon denominator',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'source_sha256':{name:sha256(Path(__file__).parent/name) for name in source_files},
        'elapsed_seconds':time.perf_counter()-started,
    }
    # Serialize before writing to avoid saving invalid/nonfinite metrics.
    serialized = json.dumps(results,indent=2,allow_nan=False)
    text = summary_text(results)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'model.npz',P=1,lam=model.lam,
                        coefficients=model.coefficients,intercept=model.intercept,
                        n_train_pairs=model.n_train_pairs,mu=data.mu,sd=data.sd,
                        log_transform=data.log_transform)
    (output/'results.json').write_text(serialized+'\n',encoding='utf-8')
    (output/'metrics_summary.txt').write_text(text,encoding='utf-8')
    if verbose:
        print('\n'+text,end='')
        print(f'\nSaved checkpoint, results and summary to {output}',flush=True)
    return model,results,output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='cave_all',help='cave_all, cave_proofread or a processed folder')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--lambdas',type=float,nargs='+',default=[0.01,0.1,1.0])
    parser.add_argument('--context',type=int,default=15)
    parser.add_argument('--horizon',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--no-log-transform',action='store_true')
    parser.add_argument('--val-frac',type=float,default=0.15)
    parser.add_argument('--test-frac',type=float,default=0.15)
    parser.add_argument('--data-kind',choices=['microns','synthetic','unspecified'],default='unspecified')
    args = vars(parser.parse_args())
    args['log_transform'] = not args.pop('no_log_transform')
    try:
        run(**args)
    except (ValueError,FileNotFoundError,FileExistsError,FloatingPointError,np.linalg.LinAlgError) as exc:
        parser.exit(1,f'VAR training: {exc}\n')


if __name__ == '__main__':
    main()
