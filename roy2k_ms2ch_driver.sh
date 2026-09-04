#!/bin/bash
# Multi-stim 2-channel (ms2ch) ladder for the 2-param (leak, kd) Roy problem.
# The CNN input = (soma volts, delivered stim) as 2 channels (--probsSelect 0 1);
# the training data spans Roy500-2000 (20k samples per family, exact rig stims).
#   M1 gen : 80k synthetic pack, per-sample stims, stim as probe 1
#   M2 base: DTW+eFEL training, 100 ep x 20k samples (stim_from_label, pad_group 64)
#   M3     : zero-shot eval on the 2ch exp pack
#   M4     : DTW fine-tune on exp Roy2000 (150 ep, 1 GPU) + eval
#   M5     : DTW fine-tune on exp Roy500-2000 (150 ep, 4 GPUs) + eval
# All stages resume (resume_checkpoint True + last.pth) and skip if their
# outputs exist -> relaunching this driver continues after any wall.
# Run under a 1-node 4-GPU allocation (sbatch roy2k_ms2ch.slr, or
# salloc -N1 -C gpu -q interactive -t 4:00:00 -A m2043_g \
#        --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
#        bash roy2k_ms2ch_driver.sh).
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
export MASTER_ADDR=$(srun -n1 hostname | head -1) MASTER_PORT=8899
echo "MASTER_ADDR=$MASTER_ADDR"

WRK=$SCRATCH/tmp_neuInv/jaxley_ca3/roy2k_ms2ch
mkdir -p $WRK
cp -rp toolbox scripts ca3_roy2k_ms2ch_dtw.hpar.yaml \
       ca3_roy2k_ms2ch_ft2000_dtw.hpar.yaml ca3_roy2k_ms2ch_ftall_dtw.hpar.yaml \
       plot_exp_overlay_royv2.py train_dist.py evaluate_voltage.py $WRK/
mkdir -p $WRK/out_ms_base $WRK/out_ms_ft2000 $WRK/out_ms_ftall
cd $WRK
PACKDIR=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_roy2k_ms2ch
EXPPACK=/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft2ch/RoyExpChaotic.mlPack1.h5
STEMS=Roy100_icav2_5k,Roy500_icav2_5k,Roy1000_icav2_5k,Roy1500_icav2_5k,Roy2000_icav2_5k

echo "=== M1: generate 80k multi-stim 2ch pack ($(date)) ==="
if [[ -f $PACKDIR/ca3_roy2k_ms2ch.mlPack1.h5 ]]; then echo "pack exists — skipping gen"; else
    if [[ -f $PACKDIR/shards/shard_003.npy ]]; then echo "shards exist — skipping worker"; else
        srun -n4 python -u scripts/gen_ca3_sharded.py \
            --cell-name ca3_roy2k_ms2ch --source-cell ca3_pyramidal \
            --out $PACKDIR --shard-dir $PACKDIR/shards \
            --stims $STEMS --per-sample-stims --fam-offset 1 \
            --n 80000 --batch 256 --vary CA3_g_leak,CA3_gkdbar_kd --trim-bins 1000
    fi
    srun -n1 python -u scripts/gen_ca3_sharded.py \
        --cell-name ca3_roy2k_ms2ch --source-cell ca3_pyramidal \
        --out $PACKDIR --shard-dir $PACKDIR/shards \
        --stims $STEMS --per-sample-stims --fam-offset 1 \
        --n 80000 --batch 256 --vary CA3_g_leak,CA3_gkdbar_kd --trim-bins 1000 \
        --merge --world 4
fi

echo "=== M2: ms2ch DTW+eFEL base training, 100 ep x 20k ($(date)) ==="
if [[ -f out_ms_base/sum_train.yaml ]]; then echo "trained — skipping"; else
    srun -n4 python -u train_dist.py --cellName ca3_roy2k_ms2ch --facility perlmutter \
        --outPath ./out_ms_base --design ca3_roy2k_ms2ch_dtw --jobId msbase_${SLURM_JOBID} \
        --probsSelect 0 1 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 100 --data_path_temp $PACKDIR/ --numGlobSamp 20000
fi

echo "=== M3: zero-shot eval on 2ch exp pack ($(date)) ==="
if [[ -f out_ms_base/exp_royv2/roy_summary.csv ]]; then echo "done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_ms_base \
        --packFile $EXPPACK --stimChannel --paramSubset 0 5
fi

echo "=== M4: DTW fine-tune on exp Roy2000, 150 ep ($(date)) ==="
if [[ -f out_ms_ft2000/sum_train.yaml ]]; then echo "trained — skipping"; else
    srun -n1 python -u train_dist.py --cellName RoyExp2000 --facility perlmutter \
        --outPath ./out_ms_ft2000 --design ca3_roy2k_ms2ch_ft2000_dtw --jobId msft2000_${SLURM_JOBID} \
        --probsSelect 0 1 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 150 --data_path_temp /pscratch/sd/k/ktub1999/RoyExp2000_ca3ft2ch/ \
        --numGlobSamp 123 \
        --do_fine_tune \
        --fine_tune-blank_model  ./out_ms_base/blank_model.pth \
        --fine_tune-checkpoint_name ./out_ms_base/checkpoints/ckpt.pth \
        --initLR 5e-5
fi
if [[ -f out_ms_ft2000/exp_royv2/roy_summary.csv ]]; then echo "M4 eval done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_ms_ft2000 \
        --packFile $EXPPACK --stimChannel --paramSubset 0 5
fi

echo "=== M5: DTW fine-tune on exp Roy500-2000, 150 ep ($(date)) ==="
if [[ -f out_ms_ftall/sum_train.yaml ]]; then echo "trained — skipping"; else
    srun -n4 python -u train_dist.py --cellName RoyExpNo100 --facility perlmutter \
        --outPath ./out_ms_ftall --design ca3_roy2k_ms2ch_ftall_dtw --jobId msftall_${SLURM_JOBID} \
        --probsSelect 0 1 --stimsSelect 0 --validStimsSelect 0 \
        --epochs 150 --data_path_temp /pscratch/sd/k/ktub1999/RoyExpNo100_ca3ft2ch/ \
        --numGlobSamp 506 \
        --do_fine_tune \
        --fine_tune-blank_model  ./out_ms_ft2000/blank_model.pth \
        --fine_tune-checkpoint_name ./out_ms_ft2000/checkpoints/ckpt.pth \
        --initLR 5e-5
fi
if [[ -f out_ms_ftall/exp_royv2/roy_summary.csv ]]; then echo "M5 eval done — skipping"; else
    srun -n1 python -u plot_exp_overlay_royv2.py -m ./out_ms_ftall \
        --packFile $EXPPACK --stimChannel --paramSubset 0 5
fi
echo "=== MS2CH DRIVER DONE ($(date)) ==="
