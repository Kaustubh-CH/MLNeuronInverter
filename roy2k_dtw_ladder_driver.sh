#!/bin/bash
# Full DTW ladder for the 2-param (leak, kd) rig-Roy2000 problem:
#   L1 base training: 80k synthetic pack, 100 ep x 20k samples, DTW+eFEL loss
#   L2 zero-shot eval on the experimental pack
#   L3 fine-tune on exp Roy2000 (150 ep, 1 GPU) + eval
#   L4 fine-tune on exp Roy500-2000 (150 ep, stim_from_label, 4 GPUs) + eval
# All stages resume (resume_checkpoint True + last.pth) and all steps skip if
# their outputs exist -> relaunching this driver continues after any wall.
# Run as:  salloc -N1 -C gpu -q interactive -t 4:00:00 -A m2043_g \
#            --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
#            bash roy2k_dtw_ladder_driver.sh
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
export MASTER_ADDR=$(srun -n1 hostname | head -1) MASTER_PORT=8888
echo "MASTER_ADDR=$MASTER_ADDR"

WRK=$SCRATCH/tmp_neuInv/jaxley_ca3/roy2k_vo2p
cp -rp toolbox ca3_roy2k_vo2p_dtw.hpar.yaml ca3_roy2k_vo2p_ft_dtw.hpar.yaml \
       ca3_roy2k_vo2p_ftall_dtw.hpar.yaml plot_exp_overlay_royv2.py \
       train_dist.py evaluate_voltage.py $WRK/
mkdir -p $WRK/out_dtw_base $WRK/out_dtw_ft2000 $WRK/out_dtw_ftall
cd $WRK
PACKDIR=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_roy2k_vo2p

echo "=== L1: DTW+eFEL base training, 100 ep x 20k ($(date)) ==="
if [[ -f out_dtw_base/sum_train.yaml ]]; then echo "trained — skipping"; else
    srun -n4 python -u train_dist.py --cellName ca3_roy2k_vo2p --facility perlmutter \
        --outPath ./out_dtw_base --design ca3_roy2k_vo2p_dtw --jobId dtwbase_${SLURM_JOBID} \
        --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 100 --data_path_temp $PACKDIR/ --numGlobSamp 20000
fi

echo "=== L2: zero-shot eval ($(date)) ==="
if [[ -f out_dtw_base/exp_royv2/roy_summary.csv ]]; then echo "done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_dtw_base --paramSubset 0 5
fi

echo "=== L3: DTW fine-tune on exp Roy2000, 150 ep ($(date)) ==="
if [[ -f out_dtw_ft2000/sum_train.yaml ]]; then echo "trained — skipping"; else
    srun -n1 python -u train_dist.py --cellName RoyExp2000 --facility perlmutter \
        --outPath ./out_dtw_ft2000 --design ca3_roy2k_vo2p_ft_dtw --jobId dtwft2000_${SLURM_JOBID} \
        --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 150 --data_path_temp /pscratch/sd/k/ktub1999/RoyExp2000_ca3ft/ \
        --numGlobSamp 123 \
        --do_fine_tune \
        --fine_tune-blank_model  ./out_dtw_base/blank_model.pth \
        --fine_tune-checkpoint_name ./out_dtw_base/checkpoints/ckpt.pth \
        --initLR 5e-5
fi
if [[ -f out_dtw_ft2000/exp_royv2/roy_summary.csv ]]; then echo "L3 eval done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_dtw_ft2000 --paramSubset 0 5
fi

echo "=== L4: DTW fine-tune on exp Roy500-2000, 150 ep ($(date)) ==="
if [[ -f out_dtw_ftall/sum_train.yaml ]]; then echo "trained — skipping"; else
    srun -n4 python -u train_dist.py --cellName RoyExpNo100 --facility perlmutter \
        --outPath ./out_dtw_ftall --design ca3_roy2k_vo2p_ftall_dtw --jobId dtwftall_${SLURM_JOBID} \
        --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 150 --data_path_temp /pscratch/sd/k/ktub1999/RoyExpNo100_ca3ft/ \
        --numGlobSamp 506 \
        --do_fine_tune \
        --fine_tune-blank_model  ./out_dtw_ft2000/blank_model.pth \
        --fine_tune-checkpoint_name ./out_dtw_ft2000/checkpoints/ckpt.pth \
        --initLR 5e-5
fi
if [[ -f out_dtw_ftall/exp_royv2/roy_summary.csv ]]; then echo "L4 eval done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_dtw_ftall --paramSubset 0 5
fi
echo "=== DTW LADDER DRIVER DONE ($(date)) ==="
