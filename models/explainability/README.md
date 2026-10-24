# Explainability: hub neurons

Tests whether a small set of recorded neurons carries a disproportionate share of predictive influence ("hubs"), and whether that influence is explained by anatomy.

-   `influence.py`: per-edge and per-neuron influence of a trained G-VAR, DCRNN or Graph WaveNet, measured by removing edges at test time
-   `hubs.py`: hub test, hub profiles, rich club and anatomy correlations on the output of `influence.py`

## Measures

-   **Per-edge influence** of j → i: the rise in neuron i's one-step test MSE when the edge is removed.
-   **Per-neuron influence** of j: (a) the sum of its per-edge influences, and (b) the rise in MSE summed over all neurons when all of j's outgoing edges are removed at once. The gap (a) − (b) is stored as `redundancy_gap`.
-   **Hub test:** the share of total per-neuron influence (b) carried by the top 20% of neurons with outgoing edges, compared with (1) the same share for out-degree and out-synapse count, and (2) the same model trained on a random graph. There is evidence for hubs when the lower bound of the 95% interval is above all of them.

MSE is in normalised units on all 15-frame test windows inside a trial. 95% intervals come from bootstrapping the 69 test trials.

## Run

```bash
# G-VAR: real graph and a degree-preserving random graph (seconds each)
python models/GVAR/train.py --dataset cave_proofread --graph connectome --horizon 1
python models/GVAR/train.py --dataset cave_proofread --graph rewired --horizon 1
python models/explainability/influence.py --model gvar --run cave_proofread_connectome
python models/explainability/influence.py --model gvar --run cave_proofread_rewired
python models/explainability/hubs.py --run gvar_cave_proofread_connectome --null gvar_cave_proofread_rewired

# DCRNN and Graph WaveNet: all steps below for both models, skipping what already exists
PYTHON=../venv-stgnn/Scripts/python.exe bash models/explainability/run_gnn.sh

# DCRNN by hand (the connectome run cave_proofread_dcrnn_h1 is already trained)
python models/stgnn/train.py --model dcrnn --dataset cave_proofread --graph rewired --run_name cave_proofread_dcrnn_rewired_h1
python models/explainability/influence.py --model dcrnn --run cave_proofread_dcrnn_h1 --stride 4
python models/explainability/influence.py --model dcrnn --run cave_proofread_dcrnn_rewired_h1 --stride 4
python models/explainability/hubs.py --run dcrnn_cave_proofread_dcrnn_h1 --null dcrnn_cave_proofread_dcrnn_rewired_h1

# Graph WaveNet, fixed adjacency only (cave_proofread_gwn_fixed_h1)
python models/stgnn/train.py --model graph_wavenet --dataset cave_proofread --graph rewired --no_learned_adjacency --run_name cave_proofread_gwn_fixed_rewired_h1
python models/explainability/influence.py --model graph_wavenet --run cave_proofread_gwn_fixed_h1 --stride 4
python models/explainability/influence.py --model graph_wavenet --run cave_proofread_gwn_fixed_rewired_h1 --stride 4
python models/explainability/hubs.py --run graph_wavenet_cave_proofread_gwn_fixed_h1 --null graph_wavenet_cave_proofread_gwn_fixed_rewired_h1
```

The GNN steps need the torch-spatiotemporal environment from `models/stgnn/requirements.txt` (torch < 2.13), not the main `venv`. `gnn_influence.ipynb` installs it and runs `run_gnn.sh` on a Kaggle or Colab GPU, saving each finished step to Drive or `/kaggle/working`.

`--graph rewired` keeps every neuron's in- and out-degree and shuffles who connects to whom, so the random graph has exactly the same wiring concentration as the connectome. `--graph random` places the same number of edges between random pairs.

G-VAR is linear, so `influence.py` computes ablations in closed form and the whole analysis takes about 15 seconds. The GNNs need one forward pass over the test set per removed edge (947 edges + 104 neurons on `cave_proofread`, 2,908 + 715 on `cave_all`). Each pass only runs on the connected component of the removed edge, since nothing outside it can change; on `cave_proofread` that is 539 of the 1,059 neurons. On a 12-core CPU with `--stride 4` this takes about 7 hours per DCRNN run and 11 hours per Graph WaveNet run. `--stride 4` uses every fourth test window, which overlap heavily anyway. `--max_edges` runs a quick test.

Outputs go to `models/explainability/runs/<model>_<run>/`: `edges.csv`, `neurons.csv`, `trial_sums.npz` (per-trial sums for the bootstrap), `meta.json`, `hubs.json` and `concentration.png`.

## Caveats

-   **G-VAR shares its coefficients across all edges**, so its edge influence is set by synapse counts, the in-degree normalisation and presynaptic activity rather than learned per edge.
-   **Removing an edge renormalises the rest.** G-VAR's shift operator and the GNNs' diffusion convolution divide by degree, so the remaining inputs of the target get more weight.
-   **The GNNs pass messages both ways.** Their diffusion convolution also uses the transposed graph, so removing j → i also cuts i → j.
-   **(b) includes indirect effects.** With K > 1 hops (G-VAR) or several message-passing steps (GNNs), removing j's edges also changes neurons further downstream. (a) only counts the direct targets, so `redundancy_gap` is not purely redundancy.
-   **Negative influence counts as zero in the shares.** Removing some edges lowers the test MSE; those neurons add nothing to the total.
-   **Graph WaveNet's learned adjacency** is not affected by removing edges, so train it with `--no_learned_adjacency`.
