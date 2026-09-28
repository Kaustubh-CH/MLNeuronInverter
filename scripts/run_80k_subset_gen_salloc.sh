#!/bin/bash -l
# 100k pack (80k train / 10k valid / 10k test) for the L5 nc2 SENSITIVE-SUBSET voltage-only run
# (2026-09-23): same recipe as the 200k pack (ncomp 2, dt 0.2, 400 ms 4k50kInterChaoticB x1.5,
# run.py sampling, LOG_HALFSPAN 1.0, fp64) but only the 10 channels that are sensitive under
# ICB x1.5 (OAT 58385397 + supervised twin R2 >= 0.64) are varied; the other 9 are pinned at the
# BBP default.  Then a GPU smoke of HybridLoss.param_subset on the new pack.
#   salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_80k_subset_gen_salloc.sh
set -u
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur || exit 2
export SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-4}
export L5TTPC_NCOMP=2 NEUINV_LOG_HALFSPAN=1.0 SOURCE_CELL=l5ttpc GEN_BATCH=64 GEN_DT=0.2
export GEN_VARY=gNaTs2_tbar_NaTs2_t_somatic,gSKv3_1bar_SKv3_1_somatic,e_pas_all,cm_somatic,gIhbar_Ih_dend,gNaTa_tbar_NaTa_t_axonal,gK_Pstbar_K_Pst_axonal,cm_axonal,gNaTs2_tbar_NaTs2_t_apical,gSK_E2bar_SK_E2_axonal
export SHARD_ROOT=/pscratch/sd/k/ktub1999/ca3_gen_shards
OUT=/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02_sub10_100k
echo "SUB10-GEN start $(date) job=${SLURM_JOB_ID:-?}"
bash scripts/run_ca3_gen.sh l5ttpc_nc2_bbp_synth "$OUT" 4k50kInterChaoticB 100000 400
echo "SUB10-GEN exit=$? $(date)"; ls -la "$OUT"
[ -f "$OUT/l5ttpc_nc2_bbp_synth.mlPack1.h5" ] || exit 1
module load python; source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=false XLA_PYTHON_CLIENT_PREALLOCATE=false
srun -n1 --gpus=1 python scripts/smoke_param_subset.py "$OUT/l5ttpc_nc2_bbp_synth.mlPack1.h5" \
     ladder_l5ttpc_nc2_icb4k_vo_fp32dt02_sub10_80k.hpar.yaml 16
echo "SUB10-SMOKE exit=$? $(date)"
