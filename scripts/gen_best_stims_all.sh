#!/bin/bash -l
# Generate all 4 "best-stim" CA3 packs (S1/S2 single, M1/M2 multi) in ONE 4-node
# GPU allocation, using the interpolated 4000-sample (_i4k, 400 ms) stims from
# sensitivity_best_stims.md. Run INSIDE `salloc -N4 --ntasks-per-node=4 ...`.
#
#   salloc -C gpu -q interactive -t4:00:00 -A m2043_g -N4 \
#          --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
#          bash scripts/gen_best_stims_all.sh
set -e
cd "$(dirname "$0")/.."
BASE=/pscratch/sd/k/ktub1999/synthetic_ca3_data
N=50000
TMAX=400

echo "############ S1: eFEL-best single (Step600) ############"
bash scripts/run_ca3_gen.sh ca3_best_efel_step600 $BASE/ca3_best_efel_step600 \
     BBP_Exp_Step600_i4k $N $TMAX

echo "############ S2: MSE-best single (Step1000) ############"
bash scripts/run_ca3_gen.sh ca3_best_mse_step1000 $BASE/ca3_best_mse_step1000 \
     BBP_Exp_Step1000_i4k $N $TMAX

echo "############ M1: eFEL-best 4-stim battery ############"
bash scripts/run_ca3_gen.sh ca3_best_efel_multi $BASE/ca3_best_efel_multi \
     5k0chaotic4_i4k,4k50kInterramp_50khz_i4k,BBP_Exp_Step600_i4k,chaotic3_i4k $N $TMAX

echo "############ M2: MSE-best 4-stim battery ############"
bash scripts/run_ca3_gen.sh ca3_best_mse_multi $BASE/ca3_best_mse_multi \
     chirp23a_i4k,BBP_Exp_Step1000_i4k,ramp_500_i4k,4k50kInterstep_500_50khz_i4k $N $TMAX

echo "############ ALL 4 PACKS DONE ############"
ls -la $BASE/ca3_best_*/ *.mlPack1.h5 2>/dev/null
for d in ca3_best_efel_step600 ca3_best_mse_step1000 ca3_best_efel_multi ca3_best_mse_multi; do
  ls -la $BASE/$d/$d.mlPack1.h5 2>/dev/null || echo "MISSING $d"
done
