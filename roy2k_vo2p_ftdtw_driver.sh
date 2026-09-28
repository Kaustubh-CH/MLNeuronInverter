#!/bin/bash
# DTW-loss variant of the stage-1 fine-tune: warm-start the same synthetic
# base model (out_vo), fine-tune on exp Roy2000 with the champion soft-DTW
# recipe (ca3_roy2k_vo2p_ft_dtw), then evaluate -> out_ft_dtw/exp_royv2.
# Run as:  salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g \
#            --ntasks-per-node=1 --gpus-per-task=1 bash roy2k_vo2p_ftdtw_driver.sh
set -e
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export MASTER_ADDR=$(srun -n1 hostname | head -1) MASTER_PORT=8887
echo "MASTER_ADDR=$MASTER_ADDR"

WRK=$SCRATCH/tmp_neuInv/jaxley_ca3/roy2k_vo2p
[[ -f $WRK/out_vo/checkpoints/ckpt.pth ]] || { echo "base out_vo missing"; exit 1; }
cp -rp toolbox ca3_roy2k_vo2p_ft_dtw.hpar.yaml plot_exp_overlay_royv2.py \
       train_dist.py evaluate_voltage.py $WRK/
mkdir -p $WRK/out_ft_dtw
cd $WRK

echo "=== FT-DTW: fine-tune on exp Roy2000 with soft-DTW loss ($(date)) ==="
if [[ -f $WRK/out_ft_dtw/sum_train.yaml ]]; then echo "trained — skipping"; else
    srun -n1 python -u train_dist.py --cellName RoyExp2000 --facility perlmutter \
        --outPath ./out_ft_dtw --design ca3_roy2k_vo2p_ft_dtw --jobId roy2kftdtw_${SLURM_JOBID} \
        --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 40 --data_path_temp /pscratch/sd/k/ktub1999/RoyExp2000_ca3ft/ \
        --numGlobSamp 123 \
        --do_fine_tune \
        --fine_tune-blank_model  ./out_vo/blank_model.pth \
        --fine_tune-checkpoint_name ./out_vo/checkpoints/ckpt.pth \
        --initLR 5e-5
fi

echo "=== FT-DTW eval ($(date)) ==="
if [[ -f $WRK/out_ft_dtw/exp_royv2/roy_summary.csv ]]; then echo "done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_ft_dtw --paramSubset 0 5
fi
echo "=== FTDTW DRIVER DONE ($(date)) ==="
