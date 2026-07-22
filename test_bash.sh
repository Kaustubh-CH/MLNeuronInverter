#!/bin/bash -l
ymlDir=/pscratch/sd/k/ktub1999/tmpYml/m8lay_vs
ymlOutDir=/pscratch/sd/k/ktub1999/tmpYmlModel
design=MultuStim
data_temp=/pscratch/sd/k/ktub1999/bbp_Jul_19_11914757/
./yaml_check.sh $ymlDir $ymlOutDir $design $data_temp &
yaml_proc_id=$!
echo $yaml_proc_id

sleep 10

kill $yaml_proc_id