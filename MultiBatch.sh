jidL="
17710054
17710128
17710172
"
# 
for jid in $jidL ; do
    dataPath=/pscratch/sd/k/ktub1999/M1_Nov_2Ex
    sbatch batchShifter.slr "$dataPath"_"$jid"
done


