#!/bin/bash
# STAGE 2 of the 2-param (leak, kd) Roy experiment ladder — run AFTER
# roy2k_vo2p_driver.sh completes:
#   S7 fine-tune the Roy2000-fine-tuned model (out_ft) on ALL experimental
#      amplitude families except Roy100 (per-sample stim via stim_from_label)
#   S8 evaluate on the full experimental test split (incl. held-out Roy100)
# Run as:  salloc -N1 -C gpu -q interactive -t 2:00:00 -A m2043_g \
#            --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
#            bash roy2k_vo2p_stage2_driver.sh
set -e
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export NCCL_DEBUG=WARN NCCL_NET_GDR_LEVEL=PHB FI_PROVIDER=cxi FI_CXI_DEFAULT_CQ_SIZE=131072
# MASTER_ADDR must be the COMPUTE node (driver runs on the login node).
export MASTER_ADDR=$(srun -n1 hostname | head -1) MASTER_PORT=8886
echo "MASTER_ADDR=$MASTER_ADDR"

WRK=$SCRATCH/tmp_neuInv/jaxley_ca3/roy2k_vo2p
[[ -f $WRK/out_ft/checkpoints/ckpt.pth ]] || { echo "stage-1 out_ft missing — run roy2k_vo2p_driver.sh first"; exit 1; }

# Refresh staged code + the stage-2 yaml (login-node copy)
cp -rp toolbox ca3_roy2k_vo2p_ftall.hpar.yaml plot_exp_overlay_royv2.py \
       train_dist.py evaluate_voltage.py $WRK/
mkdir -p $WRK/out_ftall
cd $WRK

echo "=== S7: stage-2 fine-tune on Roy500-2000 exp recordings ($(date)) ==="
if [[ -f $WRK/out_ftall/sum_train.yaml ]]; then echo "out_ftall trained — skipping"; else
    srun -n4 python -u train_dist.py --cellName RoyExpNo100 --facility perlmutter \
        --outPath ./out_ftall --design ca3_roy2k_vo2p_ftall --jobId roy2kftall_${SLURM_JOBID} \
        --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 40 --data_path_temp /pscratch/sd/k/ktub1999/RoyExpNo100_ca3ft/ \
        --numGlobSamp 506 \
        --do_fine_tune \
        --fine_tune-blank_model  ./out_ft/blank_model.pth \
        --fine_tune-checkpoint_name ./out_ft/checkpoints/ckpt.pth \
        --initLR 5e-5
fi

echo "=== S8: eval stage-2 model on full experimental pack ($(date)) ==="
if [[ -f $WRK/out_ftall/exp_royv2/roy_summary.csv ]]; then echo "done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_ftall --paramSubset 0 5
fi

echo "=== STAGE2 DRIVER DONE ($(date)) ==="
