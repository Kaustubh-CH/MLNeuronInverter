import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default='/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/L5_TTPC1cADpyr0/',help="Directory in which the jobs will store the results/ Input Directory")
    parser.add_argument("-out_dir", default='/pscratch/sd/k/ktub1999/tmp_neuInv/MeanTraining/',help="Out Put directory")
    parser.add_argument("--job-ids",default=None, type=str, nargs='+', help="list of stims, space separated")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  args=get_parser()
  loss=[]
  for jid in args.job_ids:
    directory = args.dir 
    directory_yaml= os.path.join(directory,jid)
    directory_yaml= os.path.join(directory_yaml,"out/sum_pred_nif.yaml")

    with open(directory_yaml, "r") as file:
      yaml_data = yaml.safe_load(file)
      loss.append(yaml_data["testLossMSE"])
  
  mean = np.mean(loss)
  std = np.std(loss)
  data={}
  data["Mean"]=str(mean)
  data["std"]=str(std)
  data["loss"]=loss
  data["Job IDs"]=args.job_ids
  x=range(1,len(loss)+1)
  plt.plot(x, loss, marker='o', linestyle='-')
  plt.savefig(args.out_dir+'/'+args.job_ids[0]+"loss.png")
  with open(args.out_dir+'/'+args.job_ids[0]+'.yaml', 'w') as f:
    yaml.dump(data, f)

    


