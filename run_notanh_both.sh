#!/bin/bash -l
# Driver: runs the two NO-TANH CA3 voltage-only ablations back-to-back inside a
# single salloc 4-node allocation. Each call reuses batchShifterJaxleyCA3_100ep.slr
# (100 epochs, numGlobSamp=40000) — identical to the tanh baselines except the
# design yaml has clamp_unit_tanh:False.
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter

echo "======================================================================"
echo "RUN 1/2: chaoticramp (5kChaoticRamp) NO-TANH   $(date)"
echo "======================================================================"
NEUINV_WRK_SUFIX=${SLURM_JOBID}_ct \
bash batchShifterJaxleyCA3_100ep.slr \
     /pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v1/ \
     ca3_voltonly_efel_chaoticramp_notanh || echo "RUN1 exited nonzero"

echo "======================================================================"
echo "RUN 2/2: 4kchaoticramp (4kChaoticRamp) NO-TANH  $(date)"
echo "======================================================================"
NEUINV_WRK_SUFIX=${SLURM_JOBID}_4k \
bash batchShifterJaxleyCA3_100ep.slr \
     /pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_4kchaoticramp_v1/ \
     ca3_voltonly_efel_4kchaoticramp_notanh || echo "RUN2 exited nonzero"

echo "ALL DONE $(date)"
