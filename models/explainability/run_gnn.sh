#!/usr/bin/env bash
# Influence and hub test for DCRNN and Graph WaveNet (fixed adjacency) on cave_proofread.
# Trains the degree-preserving random-graph nulls if they are missing, then runs influence.py and hubs.py.
# Steps whose output already exists are skipped, so an interrupted run can be restarted.
#
#   PYTHON=path/to/venv-stgnn/python bash models/explainability/run_gnn.sh [dcrnn] [graph_wavenet]
#
# Every removed edge or neuron is one forward pass over the test set: about 7 h per DCRNN run and 11 h per
# Graph WaveNet run on a 12-core CPU, much less on a GPU (DEVICE=cuda, the default when one is available).
# SAVE_TO=dir copies every finished step there (with the same layout as models/), e.g. Google Drive on Colab.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
STRIDE=${STRIDE:-4}
DEVICE=${DEVICE:-$($PY -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
MODELS=${*:-dcrnn graph_wavenet}

save() {  # save models/<path>
    if [ -n "${SAVE_TO:-}" ]; then
        mkdir -p "$SAVE_TO/$(dirname "$1")"
        cp -r "models/$1" "$SAVE_TO/$(dirname "$1")/"
    fi
}

for model in $MODELS; do
    case $model in
        dcrnn) tag=dcrnn; flags= ;;
        graph_wavenet) tag=gwn_fixed; flags=--no_learned_adjacency ;;
        *) echo "unknown model $model"; exit 1 ;;
    esac
    real=cave_proofread_${tag}_h1
    null=cave_proofread_${tag}_rewired_h1
    if [ ! -f models/stgnn/runs/$null/best.pt ]; then
        $PY -u models/stgnn/train.py --model $model --dataset cave_proofread --graph rewired $flags --run_name $null --device $DEVICE
        save stgnn/runs/$null
    fi
    for run in $real $null; do
        if [ ! -f models/explainability/runs/${model}_$run/meta.json ]; then
            $PY -u models/explainability/influence.py --model $model --run $run --stride $STRIDE --device $DEVICE
            save explainability/runs/${model}_$run
        fi
    done
    $PY -u models/explainability/hubs.py --run ${model}_$real --null ${model}_$null
    save explainability/runs/${model}_$real
done
