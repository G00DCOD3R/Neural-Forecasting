# G-VAR

Graph vector autoregression for one-step-ahead neural forecasting on `cave_proofread` and `cave_all`: $\mathbf{x}_t = \sum_{p=1}^{P}\sum_{k=0}^{K} h_{kp}\,\mathbf{S}^k\mathbf{x}_{t-p} + \mathbf{c}$, with $\mathbf{S}$ built from the connectome and a per-neuron intercept $\mathbf{c}$. The model is fitted in closed form (ridge), with P and K picked on validation, and compared with a copy-last baseline.

-   `data.py`: data loading, trial boundaries, train/val/test split, shift operator, exogenous inputs
-   `gvar.py`: the model
-   `train.py`: fitting, selection and evaluation
-   `test_gvar.py`: closed-form fits checked against brute-force least squares (`python -m unittest test_gvar`)
-   `gvar_forecasting.ipynb`: fits and compares all configurations

```bash
python models/GVAR/train.py --dataset cave_proofread              # graph: connectome | random, optional --P --K
python models/GVAR/train.py --P 15 --K 3 --leaky true             # LeakyReLU inputs
python models/GVAR/train.py --P 15 --K 3 --video_pcs 32 --behaviour treadmill   # G-VAR-X
python models/GVAR/train.py --clamp false --intercept false       # the original G-VAR
```

## Options

All of these keep the closed-form fit.

-   **Selection** (`--select_horizon 5`): P, K, Q and `lam_exo` are chosen on the validation MSE averaged over horizons 1..5 of the recursive forecast, the same forecast that is tested. `--select_horizon 1` selects on one-step error only.
-   **Clamp** (`--clamp`, on by default): each forecast step is clamped from below at the neuron's lowest training value before it is fed back. With the log1p transform that value is a silent neuron (raw activity 0), which is where about 79% of all entries sit.
-   **Intercept** (`--intercept`, on by default): an unpenalised per-neuron constant $c_n$.
-   **Leaky** (`--leaky`): adds taps $g_{kp}$ on $\mathbf{S}^k\,\mathrm{relu}(\mathbf{x}_{t-p})$. Together with $h_{kp}$ this is a LeakyReLU on the inputs with a learned slope per tap: $h$ below each neuron's training mean, $h + g$ above it. Because the nonlinearity acts on the inputs, the model stays linear in its parameters.
-   **Exogenous inputs, G-VAR-X** (`--video_pcs`, `--behaviour`): adds $\mathbf{B}[\mathbf{u}_t, \dots, \mathbf{u}_{t-Q+1}]$ with a per-neuron readout $\mathbf{B}$, since $\mathbf{u}$ is a global signal rather than a graph signal. $\mathbf{B}$ is eliminated per neuron in closed form, which leaves a $(K+1)P$ system for the taps. See `exogenous.py` in the repository root.
    -   Video: PCA of the stimulus frames, fitted on training rows. Run `python build_video_features.py` once to extract the frames. The stimulus is designed in advance, so frames up to the forecast target are used.
    -   Behaviour: columns of `U_exogenous.npy`, lagged by one frame and held at their last observed value during a multi-step forecast. Pupil columns contain NaNs and need `fill_missing=True` in `load_data`, which zero-fills them and adds missing-value indicators.

## Findings (cave_proofread, connectome, P=15, K=3)

Test MSE of the recursive forecast, in standardised log1p units. Validation is the selection score (mean MSE over h=1..5). G-VAR-X uses 32 video PCs and treadmill, and selected Q=1 and `lam_exo`=1.

| Model | Parameters | Validation h=1..5 | Test h=1 | Test h=5 |
|---|---:|---:|---:|---:|
| G-VAR, original (`--clamp false --intercept false`) | 60 | 0.6914 | 0.4288 | 0.7337 |
| G-VAR (clamp + intercept) | 1,119 | 0.6898 | 0.4216 | 0.7333 |
| G-VAR, leaky | 1,179 | 0.8072 | 0.4093 | 0.9776 |
| **G-VAR-X** | 36,066 | **0.6871** | 0.4216 | **0.7330** |
| G-VAR-X, leaky | 36,126 | 0.8056 | **0.4092** | 0.9757 |
| Copy-last | | | 0.6124 | 1.3311 |
| Training mean | | | 0.7930 | 0.7875 |


-   The intercept and the clamp give a small gain at h=1.
-   Leaky inputs improve one-step forecasts, but the recursive forecast drifts upwards. For a silent neuron the one-step forecast lies slightly above the floor. Fed back as an input, that value looks like the onset of activity, which the steep silent-regime slope (about 1.7 at lag 1) amplifies. Validation over five steps therefore prefers the linear model.
-   Video PCs and treadmill improve the five-step forecast slightly. On their own they explain about 1% of the validation variance; even a stimulus-locked average over 7 repeats explains only about 2.5%.
