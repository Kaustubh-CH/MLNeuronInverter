#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

jidL="
30175352

30175356

30175358

30175361

30175363

30175365

30175367

30175370

30175372

30175374

30175377

30175378

30175380

30175384

30175386

30175388

30175390

30175392

30175394

30175396

30175399

30175401

30175404

30175405

30175407

30175409

30175410

30175413

30175415

30175418

30175420

30175425

30175427

30175429

30175431

30175433

30175435

30175437

30175439

30175442

30175445

30175447

30175449

30175451

30175453

30175454

30175456

30175458

30175461
"



Path="/pscratch/sd/k/ktub1999/OntraExcRawDataAug22/runs2/" 
outPath="/pscratch/sd/k/ktub1999/BBP_Sep_19_Inh_50Cells_OntraHandPicked/"


for jid in $jidL ; do
    echo jid=$jid
    mkdir -p "$outPath"
    time  python3 aggregate_Kaustubh.py --jid ${jid}_1 --simPath $Path --outPath "$outPath" 
    k=$[ ${k} + 1 ]
done
# --idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
# --idx 0 1 2 3 4 5 6 7 8 9 10 13 14 15 18 --numExclude 4 
echo
echo SCAN: packed1 $k jobs

#python3 -m pdb aggregate_All65.py --jid 17710172 17710128 --simPath /pscratch/sd/k/ktub1999/Feb24Nrow/runs2/ --outPath /pscratch/sd/k/ktub1999/M1_ALL
