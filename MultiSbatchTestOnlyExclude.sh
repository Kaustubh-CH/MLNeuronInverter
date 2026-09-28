#!/bin/bash
# Discover the trained wrkDir for each BBP_Ontra_Exc_Mar19_NoNoise_Exclude{0..12}
# (by scanning slurm logs for a successfully-trained ni_h5_dbg run pointing
# at that data path), then submit the test-only slr in throttled batches
# of 5 (debug-queue limit).
#
# Pass indices to override the default 0..12 set:
#   bash MultiSbatchTestOnlyExclude.sh           # all 13
#   bash MultiSbatchTestOnlyExclude.sh 5 6 7     # subset

set -u

cd "$(dirname "$0")"

USER_NAME="${USER:-ktub1999}"
SLR=batchShifterTestOnlyExclude.slr
JOBNAME=ni_h5_test
BATCH_SIZE=5
DATA_PREFIX=/pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Mar19_NoNoise_Exclude
WRKDIR_ROOT=/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/ALL_CELLS

# Find a trained wrkDir for this Exclude index. Strategy:
#  1) Look at every wrkDir under $WRKDIR_ROOT
#  2) That has out/sum_train.yaml AND out/checkpoints/ckpt.pth (training completed)
#  3) Whose log.train references the matching --data_path_temp Exclude{idx}
#  4) Pick the most-recently-modified candidate (newest training run wins).
find_wrkdir_for_idx() {
    local idx=$1
    local candidate
    candidate=$(grep -lE "Exclude${idx}( |$)" $WRKDIR_ROOT/*/log.train 2>/dev/null \
              | xargs -r -I{} dirname {} \
              | while read d ; do
                  if [ -f "$d/out/sum_train.yaml" ] && [ -f "$d/out/checkpoints/ckpt.pth" ] ; then
                      printf '%s\t%s\n' "$(stat -c %Y "$d")" "$d"
                  fi
                done \
              | sort -nrk1 | head -1 | cut -f2)
    echo "$candidate"
}

# Wait until no jobs named $JOBNAME are pending or running for this user.
wait_for_test_queue_empty() {
    while true ; do
        local n
        n=$(squeue -u "$USER_NAME" -h -n "$JOBNAME" 2>/dev/null | wc -l)
        if [ "$n" -eq 0 ] ; then
            echo "[$(date '+%H:%M:%S')] no $JOBNAME jobs left in queue"
            return 0
        fi
        echo "[$(date '+%H:%M:%S')] waiting: $n $JOBNAME job(s) still pending/running"
        sleep 60
    done
}

if [ $# -gt 0 ] ; then
    ALL_IDX=("$@")
else
    ALL_IDX=(0 1 2 3 4 5 6 7 8 9 10 11 12)
fi

echo "=== Discovery: find trained wrkDir for each Exclude{idx} ==="
declare -a IDX_TODO=()
declare -a WD_TODO=()
for idx in "${ALL_IDX[@]}" ; do
    wd=$(find_wrkdir_for_idx "$idx")
    if [ -z "$wd" ] ; then
        echo "  Exclude${idx}: NO trained wrkDir found — skipping"
        continue
    fi
    echo "  Exclude${idx} -> ${wd}"
    IDX_TODO+=("$idx")
    WD_TODO+=("$wd")
done

if [ ${#IDX_TODO[@]} -eq 0 ] ; then
    echo "Nothing to submit."
    exit 0
fi

echo
echo "=== Submitting ${#IDX_TODO[@]} test-only jobs in batches of ${BATCH_SIZE} ==="
batch_num=0
i=0
while [ $i -lt ${#IDX_TODO[@]} ] ; do
    batch_num=$(( batch_num + 1 ))
    echo "================================================================"
    echo "  Test batch #${batch_num}: waiting for queue to drain before submit"
    echo "================================================================"
    wait_for_test_queue_empty
    echo "  Test batch #${batch_num} submitting at $(date)"
    submitted=0
    while [ $i -lt ${#IDX_TODO[@]} ] && [ $submitted -lt $BATCH_SIZE ] ; do
        idx="${IDX_TODO[$i]}"
        wd="${WD_TODO[$i]}"
        echo "  sbatch ${SLR} ${wd}    (Exclude${idx})"
        if sbatch "${SLR}" "${wd}" ; then
            i=$(( i + 1 ))
            submitted=$(( submitted + 1 ))
        else
            echo "  sbatch failed for Exclude${idx}; backing off 60s and retrying"
            sleep 60
        fi
    done
    echo "  batch #${batch_num} done: ${submitted} jobs submitted, $(( ${#IDX_TODO[@]} - i )) remain"
done

echo "All ${#IDX_TODO[@]} test-only jobs submitted."
