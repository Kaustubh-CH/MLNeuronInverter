#!/usr/bin/env python
'''
Not running on CPUs !
See Readmee.perlmutter

'''

import sys,os
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Trainer import Trainer 

import argparse
from pprint import pprint
import  logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
import torch
import torch.distributed as dist
from RayTune import Raytune


def get_parser():  
  parser = argparse.ArgumentParser()
  parser.add_argument("--design", default='m16lay', help='[.hpar.yaml] configuration of model and training')
  parser.add_argument("-o","--outPath", default='/pscratch/sd/k/ktub1999/tmp_neuInv/TB_logs/', type=str)
  parser.add_argument("--facility", default='perlmutter', help='data location differes')
  parser.add_argument("--cellName", type=str, default='L23_PCcADpyr2', help="cell shortName ")
  parser.add_argument("--probsSelect",default=[0,1,2], type=int, nargs='+', help="list of probes, space separated")
  parser.add_argument("--stimsSelect",default=[0], type=int, nargs='+', help="list of stims, space separated")
  parser.add_argument("--validStimsSelect",default=[0], type=int, nargs='+', help="list of stims for valid and test, space separated")
  parser.add_argument("--initLR",default=None, type=float, help="if defined, replaces learning rate from hpar")
  parser.add_argument("--epochs",default=None, type=int, help="if defined, replaces max_epochs from hpar")
  parser.add_argument("--data_path_temp",default="/pscratch/sd/k/ktub1999/bbp_May_10_8623428/", type=str, help="if defined, replaces max_epochs from hpar")
  parser.add_argument("--minLoss-RayTune",default=0.04, type=int, help="Min Threshold after which ray tune will execute 500k sample jobs")
  parser.add_argument("--minLoss-RayTune-yamlPath",default='/pscratch/sd/k/ktub1999/tmpYml', type=str,help="Path to store below min loss runs while doing Ray Tune")
  parser.add_argument("--do_fine_tune",action='store_true',default=False,help="Load a pretrained model")
  parser.add_argument("--fine_tune-blank_model",default=None, type=str,help="Path to store below min loss runs while doing Ray Tune")
  parser.add_argument("--fine_tune-checkpoint_name",default=None, type=str,help="Path to store below min loss runs while doing Ray Tune")
   
  parser.add_argument("-j","--jobId", default=None, help="optional, aux info to be stored w/ summary")
  parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
  parser.add_argument("-n", "--numGlobSamp", type=int, default=None, help="(optional) cut off num samples per epoch")


  args = parser.parse_args()
  return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == '__main__':
  torch.cuda.empty_cache()
  args = get_parser()
  if args.verb>2: # extreme debugging
      for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

  #os.environ['MASTER_PORT'] = "8879"
  #int(os.environ['SLURM_LOCALID']
  
  params ={}
  #print('M:faci',args.facility)
  if args.facility=='summit':
    import subprocess
    get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
    os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
  else:
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']

  params['world_size'] = int(os.environ['WORLD_SIZE'])
  print("Wordl Size should be 1",params["world_size"])
  verb=1
  blob=read_yaml( args.design+'.hpar.yaml',verb=1)
  params.update(blob)
  params['design']=args.design
  
  
  if(params['do_ray']):
    params['world_size']=1
    #params['world_rank'] = 0
  params['world_rank'] = 0
  # print("WORDL SIZEEEEEEEEEEEEEEEEEEEEEEee",params['world_size'])
  if params['world_size'] > 1:  # multi-GPU training
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl', init_method='env://')
    params['world_rank'] = dist.get_rank()
    #print('M:locRank:',params['local_rank'],'rndSeed=',torch.seed())
  params['verb'] =params['world_rank'] == 0

  

  if params['verb']:
    logging.info('M: MASTER_ADDR=%s WORLD_SIZE=%s RANK=%s  pytorch:%s'%(os.environ['MASTER_ADDR'] ,os.environ['WORLD_SIZE'], os.environ['RANK'],torch.__version__ ))
    for arg in vars(args):  logging.info('M:arg %s:%s'%(arg, str(getattr(args, arg))))
 
  # refine BS for multi-gpu configuration
  tmp_batch_size=params.pop('batch_size')
  if params['const_local_batch']: # faster but LR changes w/ num GPUs
    params['local_batch_size'] =tmp_batch_size 
    params['global_batch_size'] =tmp_batch_size*params['world_size']
  else:
    params['local_batch_size'] = int(tmp_batch_size//params['world_size'])
    params['global_batch_size'] = tmp_batch_size

  # capture other args values
  params['cell_name']=args.cellName
  params['data_conf']['probs_select']=args.probsSelect
  params['data_conf']['stims_select']=args.stimsSelect
  params['data_conf']['valid_stims_select']=args.validStimsSelect
  params['data_conf']['data_path']=params['data_path'][args.facility]
  params['job_id']=args.jobId
  params['out_path']=args.outPath
  params['data_path_temp']=args.data_path_temp
  params['minLoss_RayTune']=args.minLoss_RayTune
  params['minLoss_RayTune_yamlPath']=args.minLoss_RayTune_yamlPath
  params['do_fine_tune'] = args.do_fine_tune
  params['fine_tune']={}
  params['fine_tune']['blank_model'] = args.fine_tune_blank_model
  params['fine_tune']['checkpoint_name'] = args.fine_tune_checkpoint_name
  # overwrite default configuration
  #.... update selected params based on runtime config
  if args.numGlobSamp!=None:  # reduce num steps/epoch - code testing
      params['data_conf']['max_glob_samples_per_epoch']=args.numGlobSamp
  if args.initLR!=None:
        params['train_conf']['optimizer'][1]= args.initLR
  if args.epochs!=None:
        params['max_epochs']= args.epochs

  # trainer = Trainer(params)

  # trainer.train()
  if(params['do_ray']):
    rayTune = Raytune(params)
  else:
    trainer = Trainer(params)
    trainer.train()
    
  print("DONE for",params['world_rank'])
  if params['world_rank'] == 0:
    sumF=args.outPath+'/sum_train.yaml'
    write_yaml(trainer.sumRec, sumF) # to be able to predict while training continus

    print("M:done world_size=",params['world_size'])
    tp=trainer.sumRec['train_params']
    print("M:sum design=%s iniLR=%.1e  epochs=%d  val-loss=%.4f"%(tp['design'],tp['train_conf']['optimizer'][1],tp['max_epochs'],trainer.sumRec['loss_valid']))
