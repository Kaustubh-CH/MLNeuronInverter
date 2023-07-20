jidL="11914757"
dataPath=/pscratch/sd/k/ktub1999/bbp_Jul_19


# [[ -z "${NEUINV_WRK_SUFIX}" ]] && wrkSufix=$SLURM_JOBID || wrkSufix="${NEUINV_WRK_SUFIX}"

cellName=L5_TTPC1cADpyr0
wrkDir0=$SCRATCH/tmp_neuInv/bbp3//${cellName}/
outDir=$SCRATCH/tmp_neuInv/MeanTraining
# wrkDir=$wrkDir0/$wrkSufix
# /pscratch/sd/k/ktub1999/tmp_neuInv/MeanTraining


job_ids=()
for ((i=1; i<=10; i++))
do

    job_id=($(sbatch batchShifter.slr "$dataPath"_"$jidL" $jidL | awk '{print $4}'))
    job_ids+=($job_id)
done
echo "JobIDS:" 
echo ${job_ids[@]}
# scontrol --wait_job  ${job_ids[@]}

for job_id in "${job_ids[@]}"; do
    echo "Waiting for job $job_id to finish..."
    # Use scontrol to check job status. 'UNKNOWN' means the job is no longer in the queue.
    # while [ "$(scontrol show job $job_id | grep JobState | awk '{print $2}')" != "UNKNOWN" ]; do
    #     sleep 10  # Sleep for 10 seconds before checking again
    # done
    # squeue -j $job_id |grep $job_id| awk '{print $5}'
    while [ "$(squeue -j $job_id |grep $job_id| awk '{print $5}')" == "PD" ] || [ "$(squeue -j $job_id |grep $job_id| awk '{print $5}')" == "R" ] ; do
        # echo "Waiting"
        sleep 10
        
    done
    echo "Job $job_id has finished." 
done
shifter --image=balewski/ubu20-neuron8:v5 python3 toolbox/aggregate_loss.py -dir $wrkDir0  -out_dir $outDir  --job-ids ${job_ids[@]}   

# for job_id in "${job_ids[@]}"; do
#     scancel $job_id
# done

# echo "Job $job_id has finished."


