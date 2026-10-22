# Forecasting comparison protocol

I reviewed this protocol on 2026-10-22 against `main` at `f84864c`, the shared
loader in `models/common.py`, `models/baselines.py`, graph baselines in
`models/stgnn/`, and GVAR extensions at `5374dfb`. This replaces my earlier
2026-10-10 snapshot: the current shared GRU already uses whole-trial splits
and training-standardized log1p activity, rather than its original raw-activity
and row-split implementation.

## Current comparison settings

| Choice | VAR improvements | Shared GRU on main | GVAR / graph baselines |
|---|---|---|---|
| Activity | Clipped log1p; training-only per-neuron z-score | Same convention | Same convention; verify flags |
| Split | Chronological whole trials | Same convention | Same session convention |
| History | Common context 15; selected P=2 | 15 frames | GVAR order selected; STGNN history defaults to 15 |
| Inputs | Activity only; no anatomical graph | Activity only unless `--exog` | Structural/learned graph; optional exogenous inputs |
| Latest result phase | One-step validation only | Saved reports include test results | Depends on run |
| Prediction bound | Raw/bounded paired scores; bounded selection | No explicit lower bound in evaluation | Original main GVAR is unbounded; extensions can clamp; verify each run |
| Multistep | Recursive implementation; latest comparison is h=1 | Current baseline h=1 | Recursive/direct GVAR extensions; direct STGNN forecasts |

VAR learns predictive population relationships from activity; absence of an
anatomical graph does not make its neurons independent. The independent AR
control removes population inputs and fits neuron-specific coefficients.

## Alignment required before ranking models

1. Use identical timestamp-corrected activity, neuron order and trial metadata.
   Stimulus hashes identify videos, not unique presentations. Windows must
   stay inside a trial and split.
2. Fit normalization on training rows only. The shared session has 324 train /
   71 validation / 69 test trials. At context 15, h=1 has 4,602 validation and
   4,710 test windows. An h=5 test comparison has 4,434 origins instead.
3. The new controlled VAR fits use identical 21,568 training targets after
   context 15. Original VAR1 used 26,104 targets after lag 1. Report training
   origin choices even when evaluation origins match.
4. Agree the primary forecast horizon and validation selection metric before
   final test evaluation. One-step MSE and average recursive five-step MSE are
   different selection criteria. Use identical forecast origins and targets.
5. Report output bounds and whether they are fed back during recursion. Under
   clipped-log1p activity, raw zero maps to `-mu/sd` per neuron. Negative
   standardized activity is valid; do not clamp it at zero.
6. Keep activity-only and exogenous comparisons separate. Future stimulus
   frames known by design differ from future behavior unavailable at the
   forecast origin. Give compared models equivalent information access.
7. Match metric definitions, precision and constant-neuron handling. Report
   MSE, MAE, RMSE, macro Pearson and macro R2 where available. Constant pairs
   contribute zero Pearson under the epsilon convention. MSE is not accuracy.
8. Select parameters/checkpoints on validation only. The latest VAR scores
   are validation results; do not rank them against GRU/GNN test scores.

## Interpretation

- Structure and regularization changed together in the latest VAR model.
  A full-rank population component with matched separate self/population
  penalties would help isolate the rank constraint's contribution.
- The added component can contain diagonal entries, and self-history is
  updated jointly. Its gain is not a pure measure of cross-neuron information
  or causal coupling. Stored factor counts include redundancy.
- Anatomical benefit needs matched controls: GVAR K=0, randomized graphs and
  fixed versus learned graph variants. Matching random edge count alone
  does not preserve individual degrees.
- The original audit found 51 of 69 test stimulus hashes also in training.
  The split evaluates later trials, not unseen videos or another animal.
- Overlapping windows are not independent uncertainty samples. Repeated
  validation selection makes the selected score optimistic; final
  generalization claims require the agreed test evaluation.

See [current VAR validation results](var_improvements.md) and
[the historical VAR1 test report](var_results.md).
