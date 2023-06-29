jidL="
8944917
8944944
"

for jid in $jidL ; do
    dataPath=/pscratch/sd/k/ktub1999/bbp_May_18
    sbatch batchShifter.slr "$dataPath"_"$jid"
done