# Population VAR baseline

`VAR1` forecasts each neuron's activity from the previous activity of all
recorded neurons. It learns one coefficient per source-target pair and one
intercept per target, without imposing the anatomical graph. Ridge
regularization penalizes coefficients; intercepts are not penalized.

## Data and training

Run commands from the repository root using the project environment. Put the
shared timestamp-corrected session in
`processed/session_4_7_cave_proofread/`, then run:

```bash
./.venv/bin/python models/VAR/data.py --dataset cave_proofread
./.venv/bin/python models/VAR/train.py --dataset cave_proofread --data-kind microns
```

Only `X_features.npy` (finite `[time, neurons]`) and `conditions.csv` are needed.
The metadata must contain chronological `trial_id` values numbered 0..n-1 and
recording `time`, with one row per activity frame. A stimulus hash identifies
a movie; repeated presentations must have separate trial IDs. Use a folder
path with `--dataset` to load another processed session. Raw HDF5, graph files
and CAVE authentication are not needed for VAR training on processed data.

The timestamp ordering and `--out_name` changes in `build_dataset.py` originate
from [Andrei's GVAR PR #3](https://github.com/G00DCOD3R/Neural-Forecasting/pull/3)
and are carried unchanged from its reviewed head `c212580`. They are included
here because the loader requires the corrected trial metadata.

## Evaluation protocol

- Split whole trials chronologically using normalized trial start times, with
  70/15/15 time thresholds. Actual frame counts need not have those proportions.
- Clip negative activity to zero, apply `log1p`, then standardize each neuron
  with training-only mean and std + 1e-6. Validation/test use those same values.
  NaN/Inf activity is rejected. `--no-log-transform` skips clipping/log but
  retains standardization; apply any alternative consistently across models.
- Fit VAR(1) within training trials. Select lambda from `0.01 0.1 1` by
  one-step validation MSE; do not refit with validation/test targets.
- Use a common 15-frame evaluation context and five future targets. VAR(1)
  uses the last context frame. Forecasts feed predictions back recursively,
  without future observations. Windows never cross trial or split boundaries.
- Score VAR, copy-last and training-mean forecasts on identical test windows.
  Report MSE, MAE and mean per-neuron Pearson separately at each horizon.
  Constant pairs contribute zero correlation; an epsilon denominator is used.

Options include `--lambdas`, `--context`, `--horizon`, `--batch-size`,
`--output-dir`, `--val-frac` and `--test-frac`; see `train.py --help`.

## Saved results and figures

Each run creates `models/VAR/runs/run_<id>/` containing:

- `model.npz`: coefficients, intercept, selected penalty and preprocessing.
- `results.json`: configuration, validation scores, test metrics, split IDs,
  input/source SHA-256 hashes and package versions.
- `metrics_summary.txt`: a readable results table.

Nonempty output directories are not overwritten. `load_checkpoint()` in
`train.py` restores the model and preprocessing; neuron columns must retain
the training order. Data and run artifacts are ignored by Git.

Generate figures from an existing run without refitting:

```bash
./.venv/bin/python models/VAR/plot_results.py --run-dir models/VAR/runs/run_<id>
```

PNG/PDF figures are saved under that run's ignored `figures/` folder. They show
aggregate forecast metrics, rolling one-step traces and a recursive forecast
window. Examples use the first eligible test trial and fixed neuron columns
(first, middle, last). Input hashes and preprocessing are checked, and figure
metadata records the selection. `--output-dir` changes the export location.
A selected aggregate figure is included with the
[recorded MICrONS results](../../docs/var_results.md).

## Verification and examples

```bash
./.venv/bin/python -m unittest models.VAR.test_var models.VAR.test_data models.VAR.test_train -v
./.venv/bin/python models/VAR/pipeline_example.py
```

Tests check fitting against independent augmented least squares, trial
boundaries, validation-only selection, recursive forecasts, metrics,
training-only normalization, timestamp ordering and checkpoint replay.
The pipeline example uses temporary synthetic data and retains its saved run.
Synthetic scores verify the workflow and are not MICrONS performance evidence.

Additional examples cover coefficient recovery (`synthetic_example.py`), ridge
selection (`ridge_example.py`), recursive forecasts (`multistep_example.py`)
and the processed-data interface (`data_example.py`).

## Model convention and scope

For row vectors, `prediction = previous @ coefficients + intercept`, with
`coefficients[source, target]`. The objective is
`||Y - Z B - b||_F^2 / M + lambda * ||B||_F^2`, where M counts within-trial
training pairs. Fitting accumulates float64 sufficient statistics instead of
stacking all training pairs. Memory is O(N^2); the dense solve is O(N^3).
There are N*N + N fitted parameters at one lag.

Coefficients describe predictive associations. Comparing graph models requires
matched inputs, preprocessing, splits and evaluation targets; see the
[comparison protocol](../../docs/comparison_protocol.md).
