# Adaptive-GVAR

GVAR with a graph correction learned from recent neuron activity.

## Architecture

- A shared GRU reads each neuron's activity window and produces an embedding **H**. All neurons share its weights.
- Learned projections produce **Q = H W_q** and **K = H W_k**. Each row represents one neuron: Q for targets, K for sources.
- Normalize scores **Q Kᵀ / (√d × tau)** row-wise with softmax or sparsemax to get **C**. Rows are targets; columns are sources. Sparsemax allows exact zeros.
- Mix with the row-normalized anatomical graph: **S = (1 − alpha) A + alpha C**. Connections remain directed.
- GVAR uses the original activity history, powers of S, and coefficients **h[k, p]**. Everything trains together using forecasting MSE.

The saved run uses 15 history samples, graph order 3, GRU size 16, alpha 0.75, sparsemax, and tau 1.

## Reading the diagnostics

- **H:** activity embeddings from the GRU.
- **Q:** target query vectors; **K:** source key vectors.
- Cosine histograms compare neuron vectors within each window. Near 1 means similar directions; comparing H/Q/K shows the projections' effect.
- Key norm is vector length. Source attention is its C column sum: total weight received from targets. The scatter plot compares these.

In this checkpoint, median cosine similarities are H **0.75**, Q **0.63**, and K **0.90**. Keys remain more aligned than queries.

## Current results

Best checkpoint: epoch 28 of 30. One-step validation on the same windows and training-normalized log1p activity:

| Model | MSE |
|---|---:|
| Adaptive-GVAR | **0.3186** |
| GVAR (ridge) | 0.4380 |
| Copy-last | 0.6206 |
| Training mean | 0.8154 |

The correction is about **94.6% zero**, averaging **57 active incoming edges** per target across 32 sampled validation windows.

The filter heatmap shows nonzero weights across several lags and graph orders 0–3. Order 0 uses own-neuron history; higher orders use hops through the mixed graph S. Graph-based history contributes alongside own-neuron history.

## Files

- [Architecture](temporal_adaptive_gvar.py) and [tests](test_temporal_adaptive_gvar.py).
- [Training and validation](adaptive_gvar.ipynb): train or load, curves, baseline comparison.
- [Experiments](adaptive_gvar_experiments.ipynb): correction structure, H/Q/K, filter weights.
- [Checkpoint](checkpoints/adaptive_gvar_sparsemax/best.pt): saved model, configuration, training history, and optimizer state.
