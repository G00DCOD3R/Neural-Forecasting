"""Export aggregate validation results and a figure without refitting/scoring."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def export(run_dir, output_dir=None, summary_json=None):
    run_dir = Path(run_dir).resolve()
    output = Path(output_dir or run_dir/'figures').resolve()
    source = json.loads((run_dir/'selection.json').read_text())
    if source['phase'] != 'validation_selection_only' or source['config']['horizon'] != 1:
        raise ValueError('Expected a one-step validation selection run')
    chosen = source['selected']
    smallest_rank = min(row['rank'] for row in source['grid'])
    compact = min((row for row in source['grid'] if row['rank'] == smallest_rank),
                  key=lambda row: row['validation']['bounded']['mse'][0])
    rows = [source['controls']['dense_var'], source['controls']['independent_ar'], compact]
    if compact != chosen:
        rows.append(chosen)
    labels = []
    records = []
    for row in rows:
        if row['family'] == 'dense_var':
            label, name = f'Dense\nVAR({row["P"]})', f'Dense VAR({row["P"]})'
        elif row['family'] == 'independent_ar':
            label, name = f'Own-history\nAR({row["P"]})', f'Independent AR({row["P"]})'
        else:
            label = f'Diagonal +\nlow rank ({row["rank"]})\nVAR({row["P"]})'
            name = f'Diagonal-plus-low-rank VAR({row["P"]}), rank {row["rank"]}'
        labels.append(label)
        records.append({'model': name, 'family': row['family'],
                        'rank': row.get('rank'), 'stored_parameters': row['parameter_count'],
                        'lambda': row.get('lambda'), 'self_lambda': row.get('self_lambda'),
                        'population_lambda': row.get('population_lambda'),
                        'selected_iteration': row.get('selected_iteration'),
                        'metrics': {key: row['validation']['bounded'][key][0]
                                    for key in ('mse', 'rmse', 'mae', 'pearson', 'r2')}})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    colors = ['#7C879A', '#497AA0', '#69A896', '#247B68'][:len(rows)]
    for axis, key, title, minimum_limit in zip(axes, ['mse', 'mae', 'pearson'],
            ['MSE — lower is better', 'MAE — lower is better', 'Pearson r — higher is better'],
            [0.55, 0.47, 0.8]):
        values = [row['validation']['bounded'][key][0] for row in rows]
        bars = axis.bar(np.arange(len(rows)), values, color=colors, width=0.64)
        axis.set_xticks(np.arange(len(rows)), labels, fontsize=8)
        axis.set_ylim(min(0, min(values)*1.15), max(minimum_limit, max(values)*1.15))
        axis.set_title(title, fontsize=11, pad=13)
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(axis='x', length=0, pad=9)
        axis.bar_label(bars, fmt='%.4f', padding=5, fontsize=9)
    fig.suptitle('VAR validation: self-history plus a low-rank population component', fontsize=15, y=0.98)
    fig.text(0.5, 0.885,
             f'Validation only · MICrONS · {source["data"]["shape"][1]:,} neurons · one-step forecasts',
             ha='center', fontsize=10, color='#4A5568')
    windows = chosen['validation']['bounded']['n_windows']
    fig.text(0.5, 0.018,
             f'Same {windows:,} validation windows and lower prediction bound. Log1p transform; training-only standardization.\n'
             'Rank and penalty selected on validation; these are not test results. No error bars or significance claim.',
             ha='center', fontsize=9, color='#4A5568', linespacing=1.5)
    fig.subplots_adjust(top=0.77, bottom=0.27, left=0.055, right=0.985, wspace=0.27)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'var_validation_comparison.{suffix}', dpi=200, facecolor='white')
    plt.close(fig)
    summary = {'phase': source['phase'], 'source_run_id': run_dir.name,
               'control_run_id': Path(source['config']['controls_run']).name,
               'dataset': source['config']['dataset'], 'horizon': 1,
               'context': source['config']['context'], 'shape': source['data']['shape'],
               'trials_per_split': source['data']['trials_per_split'],
               'fitted_training_pairs': chosen['fitted_training_pairs'],
               'validation_windows': windows,
               'prediction_policy': 'bounded at transformed raw zero (-training_mu/training_sd)',
               'models': records, 'best_overall_family': source['best_overall_family'],
               'stopping_reason': chosen['stopping_reason'],
               'candidate_grid': {key: source['config'][key] for key in
                   ('P', 'ranks', 'population_lambdas', 'self_lambda', 'max_iter', 'patience', 'tol')},
               'input_sha256': source['data']['input_sha256'],
               'source_sha256_at_experiment': source['source_sha256'], 'versions': source['versions'],
               'scope': 'Aggregate validation summary only; no raw data, predictions, checkpoint, or test metrics.'}
    path = Path(summary_json) if summary_json else output/'var_validation_summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('Figure:', output/'var_validation_comparison.png')
    print('Aggregate results:', path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--summary-json', type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir, args.summary_json)


if __name__ == '__main__':
    main()
