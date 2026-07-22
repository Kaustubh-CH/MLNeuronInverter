#!/bin/bash
# Repack only the damaged/missing mlPack1 files in BBP_Ontra_Exc_Mar19_NoNoise_Exclude{10,12}.
# Run from inside a CPU salloc:
#   salloc -C cpu -q interactive -t4:00:00 -A m2043 -N 1
#   module load python && conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
#   bash bigDom_fix_damaged.sh
#
# After this finishes, run exclude_params_inplace.py on the two regenerated files
# so they get the same 15-column trim as the rest:
#   python3 exclude_params_inplace.py \
#     /pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Mar19_NoNoise_Exclude10/AllCellsTestOnly.mlPack1.h5 \
#     /pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Mar19_NoNoise_Exclude12/ALL_CELLS.mlPack1.h5

set -u
set -e

cd "$(dirname "$0")"

# (jid, cellName) pairs to repack.
TARGETS=(
    "10 AllCellsTestOnly"
    "12 ALL_CELLS"
)

k=0
for entry in "${TARGETS[@]}" ; do
    jid="${entry% *}"
    cell="${entry#* }"
    dataPath="/pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Mar19_NoNoise_Exclude${jid}"
    out="${dataPath}/${cell}.mlPack1.h5"
    src="${dataPath}/${cell}.simRaw.h5"

    echo "================================================================"
    echo "  jid=${jid}  cell=${cell}"
    echo "  src = ${src}"
    echo "  out = ${out}"
    echo "================================================================"

    if [ ! -f "${src}" ] ; then
        echo "  SKIP: missing source ${src}"
        continue
    fi

    # Remove damaged/old output so format script writes a fresh one.
    if [ -f "${out}" ] ; then
        echo "  removing existing ${out}"
        rm -f "${out}"
    fi

    time python3 format_bbp3_for_ML_paralelly.py --cellName "${cell}" --dataPath "${dataPath}"
    k=$(( k + 1 ))
done

echo
echo "SCAN: packed-dom ${k} jobs"
