"""Create local meeting figures from a saved VAR run; never refit the model."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'graph-ml-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from .data import load_session
    from .train import load_checkpoint, sha256
except ImportError:
    from data import load_session
    from train import load_checkpoint, sha256

COLORS = {'var': '#0072B2', 'copy_last': '#D55E00', 'mean': '#777777'}
LABELS = {'var': 'VAR', 'copy_last': 'Copy-last', 'mean': 'Training mean'}


def make_figures(run_dir, output_dir=None):
    run_dir = Path(run_dir).resolve()
    results = json.loads((run_dir / 'results.json').read_text())
    model, preprocessing = load_checkpoint(run_dir / 'model.npz')
    cfg = results['config']
    if model.lam != results['lam'] or model.parameter_count != results['parameter_count']:
        raise ValueError('Checkpoint and results do not describe the same model')
    dataset = Path(results['data']['directory'])
    for name, expected in results['data']['sha256'].items():
        if sha256(dataset / name) != expected:
            raise ValueError(f'{name} differs from the data recorded in this run')
    data = load_session(dataset, log_transform=cfg['log_transform'],
                        val_frac=cfg['val_frac'], test_frac=cfg['test_frac'])
    for name in ('mu', 'sd'):
        if not np.array_equal(getattr(data, name), preprocessing[name]):
            raise ValueError('Saved preprocessing does not match the current session')
    test_ids = np.flatnonzero(data.split == 'test')
    if test_ids.tolist() != results['data']['split_trial_ids']['test']:
        raise ValueError('Saved and current test splits differ')
    context, horizon = cfg['context'], cfg['horizon']
    trial_id = next(int(i) for i in test_ids
                    if np.diff(data.trial_bounds[i])[0] >= context + horizon)
    start, stop = data.trial_bounds[trial_id]
    trial = data.X[start:stop]
    times = data.time[start:stop]
    # Fixed column positions, chosen before looking at neuron-specific scores.
    neurons = sorted(set((0, (trial.shape[1]-1)//2, trial.shape[1]-1)))
    output = Path(output_dir).resolve() if output_dir else run_dir / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})

    def save(fig, name):
        for extension in ('png', 'pdf'):
            path = output / f'{name}.{extension}'
            fig.savefig(path, facecolor='white', bbox_inches='tight')
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    steps = np.arange(1, horizon+1)
    for name in ('var', 'copy_last', 'mean'):
        axes[0].plot(steps, results['test'][name]['mse'], 'o-',
                     color=COLORS[name], label=LABELS[name], linewidth=2)
    axes[0].set(title='Forecast error', xlabel='Forecast horizon (frames)',
                ylabel='MSE in standardized log-activity units')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc='lower right')
    axes[1].plot(steps, results['test']['var']['pearson'], 'o-',
                 color=COLORS['var'], linewidth=2)
    axes[1].set(title='Tracking activity fluctuations', xlabel='Forecast horizon (frames)',
                ylabel='Mean per-neuron Pearson correlation', ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(steps)
        ax.grid(alpha=0.2)
    fig.suptitle(f"MICrONS session 4_7 · VAR(1) · ridge {results['lam']:g}", fontsize=15)
    fig.text(0.5, 0.02,
             f"Held-out trials · {results['test']['n_windows']:,} common windows · recursive forecasts · overlapping windows",
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save(fig, 'forecast_metrics')

    # These rolling predictions observe the previous real frame at every origin.
    origins = np.arange(context, len(trial)-horizon+1)
    one_step = model.predict(trial[origins-1])
    fig, axes = plt.subplots(len(neurons), 1, figsize=(11, 7), sharex=True, squeeze=False)
    relative_times = times[origins] - times[origins[0]]
    for ax, neuron in zip(axes[:, 0], neurons):
        ax.plot(relative_times, trial[origins, neuron], color='#222222', label='Observed', linewidth=1.6)
        ax.plot(relative_times, one_step[:, neuron], color=COLORS['var'], label='VAR one-step prediction', linewidth=1.4)
        ax.set_ylabel('Standardized activity')
        ax.set_title(f'Neuron column {neuron}', loc='left', fontsize=10)
        ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, ncol=2, loc='upper right')
    axes[-1, 0].set_xlabel('Time since first shown target (seconds)')
    fig.suptitle(f'Rolling one-step forecasts · first eligible test trial {trial_id}', fontsize=14)
    fig.text(0.5, 0.015,
             'Each prediction observes the previous real population frame. Fixed columns; illustrative examples, not a population summary.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    save(fig, 'one_step_traces')

    # One common origin: all five future predictions are made before observing them.
    origin = context
    recursive = model.rollout(trial[origin-1], horizon)
    x = np.arange(-context+1, horizon+1)
    forecast_x = np.arange(0, horizon+1)
    fig, axes = plt.subplots(len(neurons), 1, figsize=(11, 7), sharex=True, squeeze=False)
    for ax, neuron in zip(axes[:, 0], neurons):
        ax.axvspan(0.5, horizon+0.1, color='#EAF3F9')
        ax.plot(x, trial[:context+horizon, neuron], 'o-', color='#222222', markersize=3,
                label='Observed activity', linewidth=1.4)
        predicted = np.r_[trial[origin-1, neuron], recursive[:, neuron]]
        ax.plot(forecast_x, predicted, 'o--', color=COLORS['var'], markersize=4,
                label='Recursive VAR forecast', linewidth=1.6)
        ax.plot(forecast_x, np.repeat(trial[origin-1, neuron], horizon+1), '--',
                color=COLORS['copy_last'], label='Copy-last', linewidth=1.2)
        ax.axvline(0.5, color='#AAAAAA', linewidth=0.8)
        ax.set_ylabel('Standardized activity')
        ax.set_title(f'Neuron column {neuron}', loc='left', fontsize=10)
        ax.grid(alpha=0.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, ncol=3, fontsize=10,
               loc='upper center', bbox_to_anchor=(0.5, 0.91))
    axes[-1, 0].set_xlabel('Frame relative to last observation (0); shaded region is future')
    axes[-1, 0].set_xticks([-context+1, -10, -5, 0, *range(1, horizon+1)])
    fig.suptitle(f'One recursive forecast window · test trial {trial_id}', fontsize=14)
    fig.text(0.5, 0.015,
             'VAR uses the last observed frame. Forecasts feed back predictions; future observations are shown only for comparison.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, 0.045, 1, 0.90))
    save(fig, 'recursive_example')

    metadata = {'run_directory': str(run_dir), 'results_sha256': sha256(run_dir/'results.json'),
                'checkpoint_sha256': sha256(run_dir/'model.npz'), 'trial_id': trial_id,
                'neuron_columns': neurons, 'recursive_origin_within_trial': origin,
                'selection_rule': 'first eligible test trial; neuron columns 0, middle, last',
                'one_step_origins': origins.tolist(), 'metric_space': results['metric_space'],
                'files': [p.name for p in paths]}
    (output/'figure_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Created figures without refitting or changing the saved run:')
    for path in paths:
        print(path)
    print('Trace examples:', f'trial {trial_id}; neuron columns {neurons}')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    make_figures(args.run_dir, args.output_dir)


if __name__ == '__main__':
    main()
