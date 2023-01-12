#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#preN=scanBS512; for lr in 0.0010 0.0015 0.0020 0.0030 0.0050  ; do 
#preN=scanBS256; for lr in 0.0007 0.0010 0.0015 0.0020 0.0030 0.0050  ; do
#preN=scanBS128; for lr in 0.0005 0.0007 0.0010 0.0015 0.0020 0.0030  ; do 
#preN=scanBS64; for lr in 0.0002 0.0003 0.0005 0.0007 0.0010 0.0015 0.0020  ; do 
#for lr in  0.0002 0.0005 0.0010 0.0020 0.0050 0.0100 ; do  #  LR
preN=scanBS512; for lr in  0.0002 0.003 0.0005 0.0007 ; do 
#for lr in 0.0002  ; do # BS512
    jobId=lr${lr}

    echo 
    echo start lr=$lr  jobId=$jobId  
    
    export NEUINV_INIT_LR=$lr
    export NEUINV_JOBID=$jobId
    export NEUINV_WRK_SUFIX=${preN}/$jobId

    #./batchShifter.slr  # interactive
    sbatch  batchShifter.slr      # Cori/PM
    #bsub  batchSummit.lsf      # Summit
    k=$[ ${k} + 1 ]
    #exit
done


echo
echo SCAN: submitted $k jobs
