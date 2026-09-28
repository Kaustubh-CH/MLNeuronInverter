#!/bin/bash
# Submit Exclude0..12 ML training jobs to the NERSC debug queue, throttled
# in batches of 5 (the per-user debug-queue limit).  Each batch is allowed
# to run to completion before the next batch is submitted.
#
# Usage: bash MultiSbatchExcludeDebug.sh

set -u

cd "$(dirname "$0")"

USER_NAME="${USER:-ktub1999}"
SLR=batchShifterDebugExclude.slr
JOBNAME=ni_h5_dbg   # SLURM -J set in $SLR; used to count this script's debug jobs
BATCH_SIZE=5
DATA_PREFIX=/pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Mar19_NoNoise_Exclude

# Wait until no jobs named $JOBNAME are pending or running for this user.
wait_for_debug_queue_empty() {
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

# Indices to submit; can be overridden by passing them as args.
if [ $# -gt 0 ] ; then
    ALL_IDX=("$@")
else
    ALL_IDX=(0 1 2 3 4 5 6 7 8 9 10 11 12)
fi
batch_num=0
i=0
while [ $i -lt ${#ALL_IDX[@]} ] ; do
    batch_num=$(( batch_num + 1 ))
    echo "================================================================"
    echo "  Debug batch #${batch_num}: waiting for queue to drain before submit"
    echo "================================================================"
    wait_for_debug_queue_empty
    echo "  Debug batch #${batch_num} submitting at $(date)"
    submitted=0
    while [ $i -lt ${#ALL_IDX[@]} ] && [ $submitted -lt $BATCH_SIZE ] ; do
        idx="${ALL_IDX[$i]}"
        data="${DATA_PREFIX}${idx}"
        echo "  sbatch ${SLR} ${data}"
        if sbatch "${SLR}" "${data}" ; then
            i=$(( i + 1 ))
            submitted=$(( submitted + 1 ))
        else
            echo "  sbatch failed for Exclude${idx}; backing off 60s and retrying same index"
            sleep 60
        fi
    done
    echo "  batch #${batch_num} done: ${submitted} jobs submitted, ${#ALL_IDX[@]}-i=$(( ${#ALL_IDX[@]} - i )) remain"
done

echo "All ${#ALL_IDX[@]} debug-queue jobs submitted."
