#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

jidL="
35503741


35503818







35503652



35503745

































35503786


35503839









35503719

35503757












35503571


35503680





35503664


35503843"

Path="/pscratch/sd/k/ktub1999/BBP_Inh_Feb5thAll150CellsNoNoise/runs2/" 
outPath="/pscratch/sd/k/ktub1999/BBP_Ontra_Inh150_Feb11_NoNoise_ExcludeHandpicked2/"

mkdir -p "$outPath"
python3  aggregate_All65.py --simPath $Path --probes 0 1 2 --fileName 'ALL_CELLS_Inhibitory_Interpolation' --outPath "$outPath"  --jid $jidL #--fileName AllCellsTestOnly 

# python3 aggregate_All65_noIndex.py --simPath $Path --outPath "$outPath" --jid $jidL --fileName 'ALL_CELLS_interpolated'
# --idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
# --idx 0 1 2 3 4 5 6 7 8 9 10 13 14 15 18 --numExclude 4 
echo
echo SCAN: packed1 $k jobs

#python3 -m pdb aggregate_All65.py --jid 17710172 17710128 --simPath /pscratch/sd/k/ktub1999/Feb24Nrow/runs2/ --outPath /pscratch/sd/k/ktub1999/M1_ALL

# shifter python3 format_bbp3_for_ML_paralelly.py --dataPath /pscratch/sd/k/ktub1999/BBP_TestOnly_Mar6
# shifter python3 format_bbp3_for_ML_paralelly.py --dataPath /pscratch/sd/k/ktub1999/ExC_Ontra_Excluded2 --cellName ALL_CELLS