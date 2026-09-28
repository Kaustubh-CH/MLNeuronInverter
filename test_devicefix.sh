#!/bin/bash -l
# Interactive 1-node, 4-task, 2-epoch smoke test to validate the GPU device-pin
# clamp (train_dist.py + Trainer.py) under --gpus-per-task=1 binding, BEFORE
# spending an 8-node batch job. Mirrors batchShifterOntraInh.slr's launch path.
set -u
cellName=ALL_CELLS_Inhibitory
design=m8lay_vs3
epochs=2
data_temp=${1:-/pscratch/sd/k/ktub1999/BBP_Ontra_Inhibitory_Exclude_NGC-DA_half}
probsSelect="0 1 2"; stimsSelect="0"; validStimsSelect="0"
facility=perlmutter

# NOTE: with `salloc ... bash thisscript`, the body runs on the LOGIN node, so
# $(hostname) would wrongly point the DDP rendezvous at the login node and hang.
# Derive MASTER_ADDR from the allocated COMPUTE node instead. (The real SLR runs
# on the compute node, so its plain `hostname` is already correct.)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export MASTER_PORT=8881
N=${SLURM_NNODES}; nprocspn=${SLURM_NTASKS_PER_NODE}; G=$(( N * nprocspn ))
echo "TEST: G=$G N=$N nprocspn=$nprocspn data=$data_temp"

wrkDir=$SCRATCH/tmp_neuInv/bbp3/${cellName}/devicefix_test_${SLURM_JOBID}
codeList="train_dist.py predict.py predictExp.py RayTune.py toolbox/ batchShifter.slr ${design}.hpar.yaml"
mkdir -p "$wrkDir/out"
cp -rp $codeList "$wrkDir"
cd "$wrkDir"

# --numGlobSamp caps samples/epoch so 4 ranks on ONE node fit in 251 GB RAM.
# (The real 8-node job has world_size=32, loads 1/32 per rank, and needs no cap.)
export CMD="python -u train_dist.py --cellName $cellName --facility $facility --outPath ./out --design $design --jobId test_$SLURM_JOBID --probsSelect $probsSelect --stimsSelect $stimsSelect --validStimsSelect $validStimsSelect --epochs $epochs --numGlobSamp 200000 --data_path_temp $data_temp"
echo "TEST CMD=$CMD"
echo "TEST wrkDir=$wrkDir"

srun -n $G shifter bash toolbox/driveOneTrain.sh
echo "TEST: srun exit=$?"
