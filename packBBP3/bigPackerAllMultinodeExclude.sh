#!/bin/bash
#!/bin/bash

jidL=(
35503991
35503993
35503995
35503996


35504015
35504018


35504020
35504022
35504023
35504024


35504027
35504029
35504032
35504034


35504042
35504043
35504045
35504046


35504061
35504064
35504068
35504069


35504071
35504072
35504073
35504074


35504076
35504077
35504078
35504079


35504081
35504082
35504083
35504084


35504087
35504088
35504089
35504090


35504092
35504093
35504094
35504095


35504099
35504101
35504102
35504103


35504110
35504113
35504116
35504118
)

# Function to submit a job with the given job IDs
submit_job() {
    local index=$1
    shift
    local job_ids=("$@")
    Path="/pscratch/sd/k/ktub1999/OntraExcRawFeb5thNoNoise/runs2/" 
    outPath="/pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Feb11_NoNoise_Exclude$index"

    mkdir -p "$outPath"
    srun -n 1 --exclusive shifter python3 aggregate_All65.py --simPath $Path --outPath "$outPath" --jid ${job_ids[*]}
    echo "Submitting job $index with IDs: ${job_ids[*]}"
    # Your job submission command here, for example:
    # qsub -v JOB_IDS="${job_ids[*]}" -v INDEX="$index" your_job_script.sh
}

total_sets=13
total_sets=5
set_size=4

for ((i=0; i<total_sets; i++)); do
    start=$((i * set_size))
    end=$((start + set_size))
    
    # Exclude the current set
    excluded_set=("${jidL[@]:start:set_size}")
    included_set=("${jidL[@]:0:start}" "${jidL[@]:end}")
    
    # Submit the job without the excluded set and pass i as an argument
    submit_job $i "${included_set[@]}"
done
wait


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
