#!/usr/bin/env bash
# Influence and hub test for DCRNN, Graph WaveNet (fixed adjacency) and Adaptive-GVAR on cave_proofread.
# Trains the degree-preserving random-graph nulls if they are missing, then runs influence.py and hubs.py.
# Steps whose output already exists are skipped, so an interrupted run can be restarted.
#
#   PYTHON=path/to/venv-stgnn/python bash models/explainability/run_gnn.sh [dcrnn] [graph_wavenet] [adaptive_gvar]
#
# Every removed edge or neuron is one forward pass over the test set: about 7 h per DCRNN run and 11 h per
# Graph WaveNet run on a 12-core CPU, much less on a GPU (DEVICE=cuda, the default when one is available).
# Adaptive-GVAR reuses its learned graphs across ablations: about 10 min per run on a CPU, and about 2 h with
# --learned (every neuron also removed from the learned graphs). Training its null takes 30 epochs; use a GPU.
# SAVE_TO=dir copies every finished step there (with the same layout as models/), e.g. Google Drive on Colab.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
STRIDE=${STRIDE:-4}
DEVICE=${DEVICE:-$($PY -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
MODELS=${*:-dcrnn graph_wavenet adaptive_gvar}

save() {  # save models/<path>
    if [ -n "${SAVE_TO:-}" ]; then
        mkdir -p "$SAVE_TO/$(dirname "$1")"
        cp -r "models/$1" "$SAVE_TO/$(dirname "$1")/"
    fi
}

influence() {  # influence <model> <run> <output folder> [extra flags]; skipped if the output exists
    if [ ! -f "models/explainability/runs/$3/meta.json" ]; then
        $PY -u models/explainability/influence.py --model "$1" --run "$2" --stride "$STRIDE" --device "$DEVICE" "${@:4}"
        save "explainability/runs/$3"
    fi
}

hubs() {  # hubs <run> <null>
    $PY -u models/explainability/hubs.py --run "$1" --null "$2"
    save "explainability/runs/$1"
}

for model in $MODELS; do
    case $model in
        dcrnn|graph_wavenet)
            if [ $model = dcrnn ]; then
                tag=dcrnn; flags=; batch=512
            else
                # 8 layers on all 1,059 neurons: 512 windows at once need more than the 15 GB of a T4
                tag=gwn_fixed; flags=--no_learned_adjacency; batch=64
            fi
            real=cave_proofread_${tag}_h1
            null=cave_proofread_${tag}_rewired_h1
            if [ ! -f models/stgnn/runs/$null/best.pt ]; then
                $PY -u models/stgnn/train.py --model $model --dataset cave_proofread --graph rewired $flags \
                    --run_name $null --device $DEVICE
                save stgnn/runs/$null
            fi
            influence $model $real ${model}_$real --batch_size $batch
            influence $model $null ${model}_$null --batch_size $batch
            hubs ${model}_$real ${model}_$null
            ;;
        adaptive_gvar)
            real=adaptive_gvar_sparsemax  # trained by models/GVAR/adaptive_gvar.ipynb on the connectome
            null=adaptive_gvar_sparsemax_rewired
            if [ ! -f models/GVAR/checkpoints/$null/results.json ]; then
                $PY -u models/GVAR/train_adaptive.py --graph rewired --run_name $null --device $DEVICE
                save GVAR/checkpoints/$null
            fi
            # anatomical edges only, as for the other models
            influence $model $real $real
            influence $model $null $null
            hubs $real $null
            # also remove each neuron from the learned graphs
            influence $model $real ${real}_learned --learned
            influence $model $null ${null}_learned --learned
            hubs ${real}_learned ${null}_learned
            ;;
        *) echo "unknown model $model"; exit 1 ;;
    esac
done
