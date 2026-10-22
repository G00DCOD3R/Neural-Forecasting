# Spatiotemporal Neural Forecasting (MICrONS)

This repository contains the data pipeline for forecasting single-neuron activity by combining functional calcium imaging with physical anatomical wiring (the connectome).

## Repository Structure

```text
Neural-Forecasting/
├── dataset/                 # (Must create manually) Folder for the raw H5 file and BossDB .pkl files.
├── processed/               # (Auto-generated) Folder containing the extracted PyTorch-ready tensors.
├── .gitignore               # Prevents massive data files from being pushed to GitHub.
├── build_dataset.py         # Unified script to extract graphs and traces into the processed folder.
├── models/baselines.py      # Copy-Last, training mean and shared GRU baselines.
├── models/VAR/              # Population VAR, independent AR and diagonal-plus-low-rank VAR.
├── example.ipynb            # Example of how to interact with the processed data
├── sanity_check.py          # Verifies the connectome graph density and overlapping neurons.
├── cave_sanity_check.py     # Same check using the full CAVE synapse table.
├── caveclient_setup.ipynb   # One-time CAVE token setup, needed for the cave_* edge sources.
├── reader.py                # Helper module to parse the Hugging Face MICrONS H5 dataset. - copied from the original dataset https://huggingface.co/datasets/NeuroBLab/MICrONS/blob/main/reader.py
├── requirements.txt         # Python dependencies needed to run the pipeline.
└── README.md                # This documentation file.           
```

## 1. Setup
Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # Or on Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

To use the CAVE connectome (`cave_sanity_check.py`, `--edge_source cave_proofread` or `cave_all`), you also need a CAVE token. Follow `caveclient_setup.ipynb` once to create and save it.

## 2. Download the Raw Data
You must place three raw data files into the dataset/ folder.

**A. The Physical Connectome (Topology)**
These files contain the verified structural mapping of synapses between neurons.
*   **What it does:** Provides the fixed physical graph adjacency matrices.
*   **Where to get it:** Download these directly from the public BossDB AWS bucket:
    *   `node_data_v1.pkl`: Download from `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/node_data_v1.pkl`
    *   `edge_data_v1.pkl`: Download from `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/edge_data_v1.pkl`
*   **Where to put it:** Save both `.pkl` files directly into your `dataset` folder.

**B. The Functional Activity (Signals)**
*   **What it does:** Contains the time-series calcium traces and behavioral variables (treadmill speed, pupil size).
*   **Where to get it:** Go to the Hugging Face `NeuroBLab/MICrONS` repository and download the `microns.h5` [file](https://huggingface.co/datasets/NeuroBLab/MICrONS/blob/main/microns.h5) (~21GB). 
*   **Where to put it:** Place `microns.h5` directly inside your `dataset` folder.


## 3. Verify the Graph (Optional but Recommended)

Before building the PyTorch tensors, run the sanity check to verify the structural graph density for our specific recording session.

```bash
python sanity_check.py
```
This should confirm that all 1,059 functional neurons have a physical counterpart. The BossDB v1 graph is very sparse (138 edges, mean in-degree ~0.13), because it only contains synapses from the 148 axons proofread for that release. Rows with `n_synapses == 0` in `edge_data_v1.pkl` are control pairs (`population` = `ADP` or `Same region`), not connections.

`cave_sanity_check.py` runs the same check against the full CAVE synapse table (requires a CAVE token, see `caveclient_setup.ipynb`). At materialization v1926 it finds 947 edges from proofread axons and 2,908 edges in total, and recovers all 138 BossDB edges.

## 4. Build the Dataset

Once the raw files are in `dataset/`, run the extraction pipeline. This script intersects the physical graph with the functional recordings and outputs PyTorch-ready tensors.
When you run the script, pass the structural session ID as `4.0` and the Hugging Face session ID as `4_7` as arguments. (Note: 4.0 refers to Session 4 in the metadata, while 4_7 refers to Session 4, Scan 7 in the H5 file, but they represent the exact same recording block).

```bash
python build_dataset.py --struct_session 4.0 --hf_session 4_7 --edge_source cave_proofread
```

`--edge_source` picks the connectome:
*   `bossdb` (default): BossDB v1 release edges with at least one synapse (138 edges).
*   `cave_proofread`: CAVE synapses whose presynaptic axon is proofread (947 edges).
*   `cave_all`: all CAVE synapses, including from unproofread axons (2,908 edges; more coverage but some false edges from segmentation errors).

Both `cave_*` sources require a CAVE token (see `caveclient_setup.ipynb`).

The CAVE version is pinned with `--mat_version` (default 1926), and the chosen source is recorded in `edge_source.txt` in the output folder.

This script intersects the physical graph with the functional recordings and will automatically create a `processed/session_4_7/` directory (or `processed/<out_name>/` with `--out_name`, e.g. `session_4_7_cave_proofread`) containing:
*   `X_features.npy` -> Shape [Timesteps, Neurons]: The target calcium signals, in recording order.
*   `U_exogenous.npy` -> Shape [Timesteps, 5]: Confounding variables (speed, pupil).
*   `edge_index.pt` -> Shape [2, Num_Edges]: Directed graph connectivity.
*   `edge_weight.pt` -> Shape [Num_Edges]: Physical synapse counts.
*   `conditions.csv` -> One row per timestep: `condition_hash` (stimulus video ID), `trial_id` (chronological trial index), `trial_name` (trial key in the H5 file) and `time` (seconds since the session started). Trials are separate stimulus presentations, with a gap of 1-2 frames between them.

## 5. Running the Baselines

To benchmark performance, execute `models/baselines.py`. It uses the shared chronological whole-trial split and training-only activity normalization:

```bash
python models/baselines.py
```
### Models Evaluated
1. **Copy-Last:** Predicts activity at timestamp $t+1$ using the observed value at timestamp $t$, establishing the naive performance floor.
2. **Training mean:** Predicts zero in training-standardized activity units.
3. **Shared GRU:** Processes 15-frame windows independently for each neuron, with one set of shared weights. Behavior inputs are disabled unless `--exog` is set. Training uses Adam, gradient clipping and validation early stopping (default patience = 6).

### Execution Outputs
Each run generates a timestamped directory `results/baselines_YYYYMMDD_HHMMSS/` containing:
* `best_gru.pt`: The checkpoint with the lowest validation loss.
* `results.json`: Training history and test metrics for GRU, Copy-Last and the training mean.

## 6. Usage in PyTorch Models
You do not need to parse the massive H5 file during model training. In your PyTorch model scripts, simply use `numpy.load` to load the `X_features.npy` and `U_exogenous.npy` files into memory. For the graph topology, use `torch.load` to load the `edge_index.pt` and `edge_weight.pt` tensors. 

From there, you can map the edge index and weights into a dense adjacency matrix for the Spatiotemporal Graph Neural Networks.

## Population VAR baseline

I developed activity-only VAR baselines that load the timestamp-corrected
processed session without raw HDF5 or graph authentication during training.
The latest comparison retains independent self-history terms and adds a
low-rank population component; it uses no anatomical graph.

```bash
./.venv/bin/python models/VAR/compare_controls.py \
  --dataset cave_proofread --lags 2 --lambdas 0.0001 0.001 0.01 0.1 \
  --context 15 --output-dir models/VAR/runs/controls_reference
```

Continue with the [diagonal-plus-low-rank workflow](models/VAR/README.md#latest-workflow-from-a-processed-session).
See [current validation results](docs/var_improvements.md),
[frozen one-step test results](docs/var_test_results.md),
[original VAR1 test results](docs/var_results.md), and the
[comparison protocol](docs/comparison_protocol.md). Recordings, checkpoints
and full run exports stay ignored; aggregate results and figures are shared.
