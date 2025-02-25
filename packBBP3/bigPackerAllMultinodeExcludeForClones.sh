#!/bin/bash
#!/bin/bash

jidL=(
35503999





35504019





35504026





35504038





35504047





35504070





35504075





35504080





35504085





35504091





35504096





35504106





35504120
)

# Function to submit a job with the given job IDs
submit_job() {
    local index=$1
    shift
    local job_ids=("$@")
    Path="/pscratch/sd/k/ktub1999/OntraExcRawFeb5thNoNoise/runs2" 
    outPath="/pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Feb11_NoNoise_Exclude$index"

    mkdir -p "$outPath"
    srun -n 1 --exclusive shifter python3 aggregate_All65.py --simPath $Path --fileName 'ALL_CELLS_interpolated' --outPath "$outPath" --jid ${job_ids[*]}&
    echo "Submitting job $index with IDs: ${job_ids[*]}"
    # Your job submission command here, for example:
    # qsub -v JOB_IDS="${job_ids[*]}" -v INDEX="$index" your_job_script.sh
}

total_sets=13
set_size=1

for ((i=0; i<total_sets; i++)); do
    start=$((i * set_size))
    end=$((start + set_size))
    
    # Exclude the current set
    excluded_set=("${jidL[@]:start:set_size}")
    included_set=("${jidL[@]:0:start}" "${jidL[@]:end}")
    
    # Submit the job without the excluded set and pass i as an argument
    submit_job $i "${included_set[@]}"
done



# jidL=(
#     22260239 22260240 22260241 22260244 22260245
#     22260247 22260248 22260249 22260251 22260252
#     22260256 22260259 22260261 22260263 22260264
#     22260266 22260267 22260269 22260271 22260272
#     22260274 22260275 22260277 22260278 22260282
#     22260284 22260285 22260289 22260291 22260293
#     22260294 22260297 22260299 22260301 22260304
#     22260305 22260306 22260311 22260313 22260315
#     22260317 22260319 22260320 22260321 22260322
#     22260323 22260325 22260326 22260327 22260328
#     22260329 22260330 22260332 22260333 22260334
# )

# # Function to submit a job with the given job IDs
# submit_job() {
#     local job_ids=("$@")
#     Path="/pscratch/sd/k/ktub1999/Jan24PaperData/runs2/" 
#     outPath="/pscratch/sd/k/ktub1999/July_24_ExC_Ontra_Excluded2actualOLD"

#     mkdir -p "$outPath"
#     srun -n 1 --exclusive shifter python3 aggregate_All65.py --simPath $Path --outPath "$outPath" --jid ${job_ids[*]}& 

#     echo "Submitting job with IDs: ${job_ids[*]}"
#     # Your job submission command here, for example:
#     # qsub -v JOB_IDS="${job_ids[*]}" your_job_script.sh
# }

# total_sets=8
# set_size=5

# for ((i=4; i<total_sets; i++)); do
#     start=$((i * set_size))
#     end=$((start + set_size))
    
#     # Exclude the current set
#     excluded_set=("${jidL[@]:start:set_size}")
#     included_set=("${jidL[@]:0:start}" "${jidL[@]:end}")
    
#     # Submit the job without the excluded set
#     submit_job "${included_set[@]}"
# done
