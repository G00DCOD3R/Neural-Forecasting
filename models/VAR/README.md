# Population VAR baselines

I implement activity-only VAR baselines without anatomical connectivity. My
latest extension keeps independent self-history terms and adds a low-rank
population component. See [validation results](../../docs/var_improvements.md) and the subsequent
[frozen one-step test report](../../docs/var_test_results.md).

## Latest workflow from a processed session

```bash
./.venv/bin/python models/VAR/compare_controls.py \
  --dataset cave_proofread --lags 2 --lambdas 0.0001 0.001 0.01 0.1 \
  --context 15 --output-dir models/VAR/runs/controls_reference

./.venv/bin/python models/VAR/select_diagonal_rank.py \
  --dataset cave_proofread --controls-run models/VAR/runs/controls_reference \
  --lags 2 --ranks 16 64 --population-lambdas 0.01 0.1 \
  --context 15 --max-iter 20 --patience 5 \
  --output-dir models/VAR/runs/diagonal_reference
```

Use new output names if those directories already contain runs. Both commands
select on one-step validation MSE and leave test forecasts unevaluated. Only
the shared processed session is needed, not raw HDF5 or a CAVE token.
The original VAR1 workflow and the other controlled experiments follow below.

`VAR1` forecasts each neuron's activity from the previous activity of all
recorded neurons. It learns one coefficient per source-target pair and one
intercept per target, without imposing the anatomical graph. Ridge
regularization penalizes coefficients; intercepts are not penalized.

## Multi-lag VAR selection

`var_p.py` adds population VAR(P), retaining the original `VAR1` implementation
and saved runs as references. Coefficients use `[lag-1, source, target]` order.
History inputs are chronological, with the oldest frame first.

```bash
./.venv/bin/python models/VAR/select_lags.py \
  --dataset cave_proofread \
  --lags 1 2 4 \
  --lambdas 0.01 0.1 1
```

This is a **training/validation-only** comparison. Every candidate fits the
same training target rows and uses the same validation target rows, after a
15-frame context. Each candidate uses only its final P context frames. This
matches the shared one-step neural-model target origins. The original VAR1 run
used additional early training targets, so this new controlled P=1 fit can
have a different score. Original results/checkpoints are preserved.

The command accumulates sufficient statistics once per lag order, then reuses
them for the ridge candidates. It selects the lowest one-step validation MSE,
records train/validation MSE, RMSE, MAE, Pearson and macro per-neuron R2, and saves
`selection.json`, `selection_summary.txt` and `model.npz` under the ignored
`models/VAR/runs/lag_selection_<id>/` directory. No test forecasts are computed.
The R2 definition matches `models/common.py`, including its epsilon denominator
and treatment of constant neurons. Metrics accumulate in float64.

All training pairs stay inside trials. At N=1,059, P=1/2/4 respectively fits
1,122,540 / 2,244,021 / 4,486,983 parameters. The dense covariance grows as
O((P*N)^2), so begin with this small lag grid. These checkpoints store a
`[P, N, N]` coefficient tensor; the original `train.py` and `plot_results.py`
remain the VAR1 workflow.

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
./.venv/bin/python -m unittest models.VAR.test_var models.VAR.test_data models.VAR.test_train models.VAR.test_var_p models.VAR.test_reduced_rank models.VAR.test_controls models.VAR.test_diagonal_low_rank models.VAR.test_evaluate_test -v
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

## Reduced-rank VAR experiment (validation only)

After selecting lag order, hold it fixed and compare reduced-rank ridge VAR
with unrestricted ridge VAR on identical fitting and validation targets:

```bash
./.venv/bin/python models/VAR/select_ranks.py \
  --dataset cave_proofread \
  --lags 2 \
  --ranks 16 32 64 128 256 512 \
  --lambdas 0.01 0.1 1 \
  --context 15
```

This fits and scores 21 candidates: six rank limits and an unrestricted
reference for each of three penalties. All use targets starting at frame 15
within training trials; validation windows use the same context. Select the
lowest one-step validation MSE. Test targets are not scored. Lag order 2 is
the winner of the preceding controlled lag comparison; this experiment tests
whether limiting rank helps at that fixed order.

The stacked coefficient matrix has shape `[P*N, N]`. Reduced rank writes it
as `input_factors @ output_factors.T`, with shapes `[P*N, r]` and `[N, r]`.
All neurons still receive predictions; `r` limits the number of shared linear
predictive patterns. The model uses activity histories and no anatomical graph.
It remains linear and retains an unpenalized per-neuron intercept.

I minimize the same mean squared error plus ridge penalty, subject to
`rank(B) <= r`. The output subspace is learned from training predictors and
targets, using the reduced-rank ridge solution of
[Mukherjee & Zhu (2011)](https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf).
This is supervised coefficient factorization, rather than PCA of the activity
or truncation of the coefficient matrix's ordinary SVD. At full output rank,
it recovers unrestricted ridge VAR. It also uses a direct solve, without epochs.

For 1,059 neurons and two lags, rank 64 stores 204,387 factor/intercept values
versus 2,244,021 dense coefficients/intercepts. Factor counts include rotational
redundancy. The separately recorded `coefficient_dimension` subtracts that
redundancy; neither count measures ridge's statistical effective degrees of
freedom. Fitting still forms dense sufficient statistics and solves the full
ridge system, reusing each fit and rank decomposition across candidate ranks.
Factorization reduces prediction/storage cost; it does not make this fitting
algorithm sparse or remove its dense memory requirements.

Outputs go into ignored `runs/rank_selection_<id>/`: `selection.json`,
`selection_summary.txt`, and the selected `model.npz`. The checkpoint stores
factors for a reduced-rank winner or dense coefficients for an unrestricted
winner, plus intercept, preprocessing and fitting metadata. Use
`load_checkpoint()` in `select_ranks.py` for these checkpoints; the original
VAR(1) plotting/checkpoint loader does not support this format.
The numerical checks compare against an independent augmented QR/SVD solution,
check full-rank equivalence and recursive forecasts, and verify that changing
test activity cannot change validation selection. Improved MICrONS accuracy
is an experimental question, not guaranteed by the constraint.

## Matched VAR controls and prediction bound (validation only)

Before extending the factorized model, measure how much dense VAR gains from
other neurons beyond each target's own history:

```bash
./.venv/bin/python models/VAR/compare_controls.py \
  --dataset cave_proofread \
  --lags 2 \
  --lambdas 0.01 0.1 1 \
  --context 15
```

This fits six models: dense VAR(2) and independent AR(2), each at three ridge
penalties. Independent AR is the diagonal restriction of VAR: every neuron
has its own P self-history coefficients and intercept, with cross-neuron
coefficients fixed to zero. It is a diagnostic control for the population VAR
task, not a graph model or a replacement for the dense reference. At 1,059
neurons and two lags it has 3,177 parameters versus 2,244,021 in dense VAR.
Both families fit exactly the same 21,568 training targets for this session,
using the same ridge objective and unpenalized-intercept convention.

The script reports both unbounded and bounded one-step forecasts on identical
origins. Clipping raw activity to zero before log1p implies a known target
lower bound `floor = -training_mu / training_sd` per neuron. Standardized
values below zero remain valid; predictions are bounded at this per-neuron
floor, not at zero. This convention follows the target transform and does
not assert that all raw measurement values are nonnegative. The helper rejects
the no-log-transform setting, for which this bound is not justified.

Select the penalty separately within each family by **bounded one-step
validation MSE**. The unbounded columns preserve a comparison with the earlier
lag/rank experiments, which did not use this prediction policy. Scores from
the new policy must be labeled accordingly. The script does not select lag
order again or score test targets; it also reports MAE, RMSE, Pearson and macro
per-neuron R2. A validation difference is not itself a significance test or a
causal measure of coupling.

`BoundedForecast` applies the bound before each recursive feedback step. For
fixed one-step predictions and targets in the valid range, applying the bound
cannot increase squared error. That guarantee does not extend automatically
to recursively evaluated later horizons. The command above selects and
evaluates only horizon 1; it makes no multistep performance claim.

The ignored `runs/controls_<id>/` folder contains `selection.json`,
`selection_summary.txt` and separate selected `dense_var.npz` and
`independent_ar.npz` checkpoints. These record the original fitted weights,
preprocessing and prediction floor. `load_checkpoint()` in
`compare_controls.py` restores the bounded forecast; `.base` exposes its
underlying linear model. Use that loader for these checkpoint formats.
The comparison reuses dense sufficient statistics for both fits; its overall
fitting memory therefore still grows quadratically with the population size.

Tests check the independent fit against separate augmented least-squares
problems, enforce isolation from other neurons, verify correct bound feedback,
replay both saved models, and confirm that changing test activity cannot
affect selection. Existing dense and reduced-rank scripts/results retain their
original behavior. Diagonal-plus-low-rank VAR is a separate experiment,
described below; it is not implemented by this control command.

## Diagonal-plus-low-rank VAR (validation only)

Use the verified controls run from the preceding step:

```bash
./.venv/bin/python models/VAR/select_diagonal_rank.py \
  --dataset cave_proofread \
  --controls-run models/VAR/runs/controls_reference \
  --lags 2 \
  --ranks 16 64 \
  --population-lambdas 0.01 0.1 \
  --context 15 \
  --max-iter 20 \
  --patience 5
```

Replace the control-run path if using another session/run. Input and source
hashes, preprocessing, trial IDs, lag order, context, fitting row counts and
validation replay must match. No test forecasts are computed.

The stacked VAR coefficient matrix is `B = E(d) + A @ V.T`, where `E(d)` stacks
P diagonal self-history matrices and the added component has rank at most r.
Every neuron retains its own P self-history coefficients; only the added
component is rank constrained. It may also contain diagonal entries: forcing
those entries to zero would change the conditional rank-constrained problem.
The total coefficient matrix is allowed to have full output rank.
This is a population VAR extension using no anatomical graph or external
features. Factor/diagonal contributions are not causal attributions.

The structure is motivated by
[Wagenmaker et al., NeurIPS 2024](https://arxiv.org/html/2412.02529v4#S3.SS1).
I adapt this structure to a different task: visual activity forecasting.
I constrain the stacked population component and use the regularized solver
below, rather than reproduce their per-lag factor fitting procedure.

Fit `SSE/M + self_lambda*||d||^2 + population_lambda*||A @ V.T||_F^2`, with an
unpenalized intercept. Here M counts common within-trial training target rows.
The factor product is penalized, rather than the separate factor norms.
Self-history regularization is fixed to the selected independent AR penalty
from the control run. The small pilot grid above has four candidates.

Iteration 0 reproduces independent AR, with zero population factors. Each
iteration then solves two training-only conditional problems exactly: a
reduced-rank ridge fit of the population component given self-history, then
per-neuron ridge fits of self-history given the population component. The
training objective must not increase. For readable units the recorded
objective divides the entire penalized criterion by the neuron count.
Fitting still uses dense sufficient statistics, a dense factorization and a
rank decomposition; this is not a sparse-memory training algorithm.

The joint structured problem is nonconvex, so these iterations do not guarantee
a global optimum. Validation chooses the best complete iterate, including
iteration 0. Stop after the iteration cap, training-objective tolerance, or
the specified number of iterations without validation improvement. An early
stop is not proof that all possible fits of this model have been exhausted.
The bound is applied for validation prediction, not inserted into the raw
linear ridge fitting loss. Both raw and bounded validation metrics are kept.

At 1,059 neurons and P=2, rank 16 stores 54,009 values and rank 64 stores
206,505 values, counting self-history, both factors and intercept. These counts
include factor redundancy and are not statistical effective degrees of freedom.

Outputs are saved to ignored `runs/diagonal_rank_<id>/`: `selection.json`,
`selection_summary.txt` and `model.npz`. The JSON includes both verified
controls, iteration histories, stopping reasons, selected iterates and the
best overall model family. The checkpoint contains the best added-model
candidate even if a control wins overall; iteration 0 means its population
component did not improve on the independent control. Restore it with
`load_checkpoint()` in `select_diagonal_rank.py`, which returns the bounded
forecast and preprocessing. Existing scripts do not read this checkpoint format.

Checks compare each conditional update against independent augmented
least-squares/SVD calculations, compare the unrestricted-population case
against a joint ridge solve, verify the full training objective directly from
data and its monotonic descent, and exercise saved-model replay and validation
selection without test influence.

Export a figure and aggregate results from a completed run without refitting:

```bash
./.venv/bin/python models/VAR/plot_validation.py \
  --run-dir models/VAR/runs/diagonal_reference
```

The default export location is that ignored run's `figures/` directory. Use
`--output-dir` and `--summary-json` for explicit shareable export locations.

## Frozen one-step test evaluation

After fixing a configuration from validation, score its saved checkpoint and
both selected controls without refitting or changing parameters:

```bash
./.venv/bin/python models/VAR/evaluate_test.py \
  --dataset cave_proofread \
  --run-dir models/VAR/runs/diagonal_reference \
  --controls-run models/VAR/runs/controls_reference
```

The evaluator verifies source/input fingerprints, checkpoint settings and
training-only preprocessing, and replays the recorded validation scores. It
writes `frozen_protocol.json` before computing test forecasts, then scores
all three frozen models and copy-last/mean on identical one-step test windows.
The bounded prediction policy was selected during validation; unbounded
scores are a predeclared diagnostic, not another selection criterion.

The ignored `runs/test_one_step_<id>/` folder contains the frozen protocol,
`results.json` and `metrics_summary.txt`. There are no rank, penalty or horizon
selection flags. Nonempty output directories are not overwritten. See the
[test report](../../docs/var_test_results.md) for the recorded results and
comparison limits. These test scores must not drive further tuning of this
frozen comparison. A future five-step task requires a separately agreed protocol.
