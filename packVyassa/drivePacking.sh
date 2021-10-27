#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;    #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value

echo 'packing-driver start on '`hostname`' '`date`

nameList=$1
cloneId=$2
packExt=$3
mkdir -p out

procIdx=${SLURM_PROCID}
maxProc=`cat $nameList |wc -l`

echo my procIdx=$procIdx   maxProc=$maxProc "pwd: "`pwd`  
sleep $(( 1 + ${SLURM_LOCALID}*30  + $SLURM_PROCID))
echo RANK_`hostname` $SLURM_PROCID $nameList

if [ $procIdx -ge $maxProc ]; then
   echo "rank $procIdx above maxProc=$maxProc, idle ..."; exit 0
fi

line=`head -n $[ $procIdx +1 ] $nameList |tail -n 1`
#line="bbp205 L6_TPC_L1_cADpyr231 48810123_78 x"
echo line=${line}=

bbpNNN=`echo $line | cut -f1 -d\ `
bbpBase=`echo $line | cut -f2 -d\ `
simuJobIdS=`echo $line | cut -f3 -d\ `

cmd=" ./postproc_oneSimu.sh  $bbpNNN  $bbpBase  $simuJobIdS  $cloneId $packExt "

echo cmd=$cmd=
bbpNNNC=${bbpNNN}$cloneId
eval "$cmd" >& out/log.$procIdx.$bbpNNNC

echo "$bbpNNNC task_done  on  "`date` 
