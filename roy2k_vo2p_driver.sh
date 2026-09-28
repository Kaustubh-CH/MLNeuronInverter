#!/bin/bash
# Two-experiment session: 2-param (leak, kd) voltage-only CA3 model under the
# EXACT rig Roy2000 stimulus.
#   S1 generate 80k synthetic traces (gen_ca3_sharded, --vary leak,kd, trim 1000)
#   S2 voltage-only training (4 GPUs, 15 ep)
#   S3 predict.py on the synthetic test split (param R2 for leak/kd)
#   S4 Exp A: zero-shot predict on the experimental pack (voltage error)
#   S5 Exp B: fine-tune on Roy2000-family experimental recordings (1 GPU)
#   S6 eval the fine-tuned model the same way
# Every step is skipped if its output already exists -> re-running the driver
# resumes after an allocation timeout.
# Run as:  salloc -N1 -C gpu -q interactive -t 4:00:00 -A m2043_g \
#            --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
#            bash roy2k_vo2p_driver.sh
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
# MASTER_ADDR must be the COMPUTE node — this driver runs on the login node,
# so resolve it with a trivial srun step (hostname of the allocated node).
export MASTER_ADDR=$(srun -n1 hostname | head -1) MASTER_PORT=8885
echo "MASTER_ADDR=$MASTER_ADDR"

PACKDIR=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_roy2k_vo2p
PACK=$PACKDIR/ca3_roy2k_vo2p.mlPack1.h5
SHARDS=$PACKDIR/shards
WRK=$SCRATCH/tmp_neuInv/jaxley_ca3/roy2k_vo2p
EXPPACK2000=/pscratch/sd/k/ktub1999/RoyExp2000_ca3ft/

# Stage from the LOGIN node (compute-node /global/u1 reads are minutes-slow)
mkdir -p $WRK/out_vo $WRK/out_ft $WRK/scripts
cp -rp train_dist.py predict.py evaluate_voltage.py plot_exp_overlay_royv2.py \
       toolbox ca3_roy2k_vo2p.hpar.yaml ca3_roy2k_vo2p_ft.hpar.yaml $WRK/
cp -p scripts/gen_ca3_sharded.py $WRK/scripts/
echo "staged $WRK"
cd $WRK

echo "=== S1: generate 80k under Roy2000_icav2_5k, vary leak+kd ($(date)) ==="
if [[ -f $PACK ]]; then echo "pack exists — skipping"; else
    mkdir -p $SHARDS
    srun -n4 python -u scripts/gen_ca3_sharded.py --cell-name ca3_roy2k_vo2p \
        --out $PACKDIR --shard-dir $SHARDS --stims Roy2000_icav2_5k \
        --n 80000 --t-max 500 --vary CA3_g_leak,CA3_gkdbar_kd --trim-bins 1000
    srun -n1 python -u scripts/gen_ca3_sharded.py --cell-name ca3_roy2k_vo2p \
        --out $PACKDIR --shard-dir $SHARDS --stims Roy2000_icav2_5k \
        --n 80000 --t-max 500 --vary CA3_g_leak,CA3_gkdbar_kd --trim-bins 1000 \
        --merge --world 4
fi

echo "=== S2: voltage-only training, 100 epochs x 20k samples, 4 GPUs ($(date)) ==="
# 20k samples/epoch ~2.7 min/epoch on one node -> 100 ep ~4.5 h: crosses the
# 4 h wall.  resume_checkpoint:True + last.pth means a driver relaunch resumes
# at the epoch it died on and then continues with S3-S6.
if [[ -f $WRK/out_vo/sum_train.yaml ]]; then echo "out_vo trained — skipping"; else
    srun -n4 python -u train_dist.py --cellName ca3_roy2k_vo2p --facility perlmutter \
        --outPath ./out_vo --design ca3_roy2k_vo2p --jobId roy2kvo_${SLURM_JOBID} \
        --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 100 --data_path_temp $PACKDIR/ --numGlobSamp 20000
fi

echo "=== S3: synthetic test param R2 (predict.py) ($(date)) ==="
if [[ -f $WRK/out_vo/sum_pred_nif.yaml ]]; then echo "predicted — skipping"; else
    srun -n1 python -u predict.py --modelPath ./out_vo || echo "WARN: predict.py failed (non-fatal)"
fi

echo "=== S4: Exp A — zero-shot on experimental pack ($(date)) ==="
if [[ -f $WRK/out_vo/exp_royv2/roy_summary.csv ]]; then echo "done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_vo --paramSubset 0 5
fi

echo "=== S5: Exp B — fine-tune on Roy2000 experimental recordings ($(date)) ==="
if [[ -f $WRK/out_ft/sum_train.yaml ]]; then echo "out_ft trained — skipping"; else
    srun -n1 python -u train_dist.py --cellName RoyExp2000 --facility perlmutter \
        --outPath ./out_ft --design ca3_roy2k_vo2p_ft --jobId roy2kft_${SLURM_JOBID} \
        --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 40 --data_path_temp $EXPPACK2000 --numGlobSamp 123 \
        --do_fine_tune \
        --fine_tune-blank_model  ./out_vo/blank_model.pth \
        --fine_tune-checkpoint_name ./out_vo/checkpoints/ckpt.pth \
        --initLR 5e-5
fi

echo "=== S6: eval fine-tuned model on experimental pack ($(date)) ==="
if [[ -f $WRK/out_ft/exp_royv2/roy_summary.csv ]]; then echo "done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_ft --paramSubset 0 5
fi

echo "=== DRIVER DONE ($(date)) ==="
