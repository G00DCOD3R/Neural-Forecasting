# Forecasting comparison protocol

Proposed settings for comparing population VAR, GVAR and GRU. The group still
needs to agree on the shared protocol before ranking all models.

Reviewed 2026-10-10: [GVAR PR #3](https://github.com/G00DCOD3R/Neural-Forecasting/pull/3)
at `c212580`, and [GRU/copy-last PR #2](https://github.com/G00DCOD3R/Neural-Forecasting/pull/2)
at `896e5b3`. Branch settings may change after this review.

## Current settings

| Choice | Population VAR | GVAR PR #3 | GRU PR #2 |
|---|---|---|---|
| Activity scaling | Clipped log1p + training z-score | Same | Raw |
| Split/boundaries | Explicit trial IDs; whole-trial time split | Same | Row split; stimulus-hash endpoint check |
| Activity history | One lag | Validation grid, up to 15 lags | Ten frames |
| External inputs | None | None | Behavior history |
| Evaluation | 15-frame context; five recursive steps | Same | Ten-frame context; one step |
| Constant-neuron Pearson | Included as zero | Same intended convention | Constant pairs skipped |
| Parameters | N*N + N, including neuron intercepts | P*(K+1), shared coefficients, no intercept | Shared GRU and output layer |
| Validation selection | Ridge penalty | Temporal/graph orders P/K; fixed ridge | Best validation epoch |

On the shared session, VAR and GVAR produce identical normalized activity,
trial IDs and split labels, with 4,602 one-step validation origins and 4,434
five-step test origins. A first-test-trial check confirmed matching baseline
MSE/MAE within float32 tolerance. Pearson differed by up to 0.000337 due to
numerical precision; all baseline metrics agreed within 1e-10 when GVAR's
calculation used float64. Use consistent precision and constant-neuron handling
when interpreting small metric differences.

## Choices to agree on

1. Use the same timestamp-corrected session, neuron order and explicit trial IDs.
   Stimulus hashes identify movies, not unique presentations; endpoint equality
   alone does not establish that a whole window stays inside one trial.
2. Use training-only normalization and whole-trial chronological splits. The
   current VAR/GVAR settings provide a working reference. Raw-space GRU MSE
   cannot be compared directly with normalized VAR/GVAR MSE.
3. Establish an activity-only comparison. If behavior inputs are included,
   label that extension and give compared models equivalent input access.
4. Use identical validation and test target origins. A five-step comparison
   requires the same future-target availability and recursive input policy.
   A separate one-step comparison is possible before extending GRU.
5. Choose settings using validation only, and report lag orders, parameter
   counts and selection procedures. Numerical ridge penalties are not
   interchangeable between these parameterizations.
6. Include GVAR K=0 and randomized-graph controls. Current randomization preserves
   edge count and shuffles weights, but does not preserve individual degrees.
   Multiple random seeds would strengthen this control.
7. State the generalization target: 51 of 69 test stimulus IDs also occur in
   training. This split evaluates later trials, not unseen movies. Forecast
   windows overlap, so they are not independent samples for uncertainty estimates.

## Preliminary GVAR controls

Andrei's saved notebook reports the following rounded results; these GVAR
experiments were not rerun for this comparison. Confirm exact run provenance
before combining them into the final report.

| Model | Lags P | Graph order K | h=1 MSE | h=5 MSE |
|---|---:|---:|---:|---:|
| Population VAR | 1 | Not applicable | 0.496943 | 0.733091 |
| GVAR, proofread connectome | 15 | 3 | 0.4288 | 0.7337 |
| GVAR, no graph | 15 | 0 | 0.4292 | 0.7341 |
| GVAR, proofread random graph | 15 | 3 | 0.4291 | 0.7340 |

The recorded anatomical benefit within GVAR is small. Its larger difference
against VAR(1) cannot be attributed solely to wiring because temporal history
and parameterization also differ. A validation-only VAR lag search is a
possible follow-up after the shared comparison protocol is agreed.
