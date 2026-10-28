# Heatmaps of the graph-filter taps h_kp (and g_kp for leaky models) of saved G-VAR runs, one row per run.
#   python models/GVAR/plot_taps.py cave_proofread_connectome cave_proofread_connectome_nl2_rollout_p6
# Taps are read from model.pt, since results.json holds the closed-form taps from before any gradient training.
# Direct models get one panel per horizon. Each row shares one symmetric colour scale.
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

RUNS_DIR = Path(__file__).resolve().parent / 'runs'
# h / g of GVAR, heads.j.h of DirectGVAR, both under linear. inside NonlinearGVAR
TAP_KEY = re.compile(r'^(?:linear\.)?(?:heads\.(\d+)\.)?([hg])$')


def load_taps(run):
    state = torch.load(RUNS_DIR / run / 'model.pt', map_location='cpu')
    taps = []
    for key, value in state.items():
        m = TAP_KEY.match(key)
        if m:
            head, name = m.groups()
            label = f'${name}$' + (f', horizon {int(head) + 1}' if head is not None else '')
            taps.append((int(head or 0), name, label, value.numpy()))
    return [(label, v) for _, _, label, v in sorted(taps, key=lambda t: (t[0], t[1]))]


# labels: row titles, the run names by default
def plot(runs, labels=None):
    rows = list(zip(labels or runs, map(load_taps, runs)))
    cols = max(len(t) for _, t in rows)
    fig, axes = plt.subplots(len(rows), cols, figsize=(4.2 * cols, 2.6 * len(rows)), squeeze=False)
    for ax_row, (run, taps) in zip(axes, rows):
        lim = max(np.abs(v).max() for _, v in taps)
        for j, (ax, (label, v)) in enumerate(zip(ax_row, taps)):
            im = ax.imshow(v, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
            ax.set_xticks(range(v.shape[1]), range(1, v.shape[1] + 1), fontsize=7)
            ax.set_yticks(range(v.shape[0]))
            # the row title only above its first panel
            ax.set(title=(f'{run}\n' if j == 0 else '\n') + label, xlabel='lag p', ylabel='graph shift k')
            ax.title.set(fontsize=8, ha='left' if j == 0 else 'center', x=0 if j == 0 else 0.5)
            ax.grid(False)
        fig.colorbar(im, ax=ax_row[len(taps) - 1])
        for ax in ax_row[len(taps):]:
            ax.axis('off')
        for label, v in taps:
            print(f'{run:50s} {label:12s} sum over lags per k:', np.round(v.sum(1), 3))
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='run directories under models/GVAR/runs')
    parser.add_argument('--labels', nargs='+', default=None, help='row titles, one per run')
    parser.add_argument('--out', type=Path, default=RUNS_DIR / 'taps.png')
    args = parser.parse_args()
    plot(args.runs, args.labels).savefig(args.out, dpi=200)
    print(f'Saved {args.out}')
