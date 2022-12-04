#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#for lr in  0.0002 0.0005 0.0010 0.0020 0.0050 0.0100 ; do  #  LR - Cori/PM/Summit
for lr in 0.007 ; do # one-off
#for lr in   0.0005 0.0010 0.0020 0.0050 0.0100 ; do  #  LR - Cori/PM/Summit
    jobId=lr${lr}

    echo 
    echo start lr=$lr  jobId=$jobId  
    
    export NEUINV_INIT_LR=$lr
    export NEUINV_JOBID=$jobId
    export NEUINV_WRK_SUFIX=$jobId
  
    sbatch  batchShifter.slr      # Cori/PM
    #bsub  batchSummit.lsf      # Summit
    k=$[ ${k} + 1 ]
    #exit
done


echo
echo SCAN: submitted $k jobs
