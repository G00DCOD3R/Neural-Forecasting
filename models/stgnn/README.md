# DCRNN and Graph WaveNet baselines

Reference graph forecasters from [torch-spatiotemporal](https://torch-spatiotemporal.readthedocs.io/). They use the same processed session, trial-level split, `log1p` transform and training-set normalisation as G-VAR. The default context is 15 frames, matching the G-VAR evaluation windows. Forecasts are direct, not rolled out.

Behaviour is excluded unless `--exog` is set, so a run is comparable with G-VAR. The graph is the saved connectome (`edge_index.pt`, presynaptic to postsynaptic, weighted by synapse count). `--graph random` keeps the same edge count and shuffled weights.

Install into the project virtual environment, as described in the repository README. Python 3.13 is required.

```bash
source venv/bin/activate
python models/stgnn/train.py --smoke
python models/stgnn/train.py --model dcrnn --dataset cave_proofread
python models/stgnn/train.py --model graph_wavenet --dataset cave_proofread
python models/stgnn/train.py --model graph_wavenet --dataset cave_proofread --no_learned_adjacency
```

`--no_learned_adjacency` keeps only the anatomical graph. The default Graph WaveNet adds its learned adaptive adjacency on top of that graph. Results go to `models/stgnn/runs/<run>/results.json` and include MSE, RMSE, MAE, Pearson correlation and R², plus copy-last and training-mean baselines on the same test windows.
