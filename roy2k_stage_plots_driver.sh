#!/bin/bash
# Validation-domain evals + all-stage voltage-trace overlays for the 2-param
# ladder (zero-shot / ft2000 / ftall / ft_dtw), test AND valid domains.
# Run as:  salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g \
#            --ntasks-per-node=1 --gpus-per-task=1 bash roy2k_stage_plots_driver.sh
set -e
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

WRK=$SCRATCH/tmp_neuInv/jaxley_ca3/roy2k_vo2p
cp -p plot_stage_overlays.py plot_exp_overlay_royv2.py $WRK/
cd $WRK

for m in out_vo out_ft out_ftall out_ft_dtw; do
    [[ -f $m/sum_train.yaml ]] || { echo "skip $m (no sum_train)"; continue; }
    if [[ -f $m/exp_royv2_valid/roy_summary.csv ]]; then echo "$m valid eval done"; else
        echo "=== valid eval: $m ($(date)) ==="
        srun -n1 python -u plot_exp_overlay_royv2.py -m ./$m --dom valid \
             --outDir ./$m/exp_royv2_valid --paramSubset 0 5
    fi
done

MODELS="zero-shot=./out_vo ft2000=./out_ft ftall=./out_ftall ft_dtw=./out_ft_dtw"
for dom in test valid; do
    echo "=== stage overlays: $dom ($(date)) ==="
    srun -n1 python -u plot_stage_overlays.py --dom $dom --models $MODELS \
         --outDir ./stage_overlays
done
echo "=== STAGE PLOTS DRIVER DONE ($(date)) ==="
