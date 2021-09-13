#!/usr/bin/env python
""" 
read trained net : model+weights
read test data from HD5
infere for  test data 

Inference works always on 1 GPU
srun -n1 ./predict.py

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import torch
from pprint import pprint
import  time
import sys,os
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Plotter import Plotter_NeuronInverter
from toolbox.Dataloader_H5 import get_data_loader
from toolbox.Util_H5io3 import read3_data_hdf5,write3_data_hdf5
from predict import load_model

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--facility", default='corigpu', type=str)
    parser.add_argument("-m","--modelPath",  default='/global/cscratch1/sd/balewski/tmp_digitalMind/neuInv/manual/', help="trained model ")
    parser.add_argument("-d","--dataPath",  default='exp4ml/', help="exp dat for ML pred")
    parser.add_argument("-o", "--outPath", default='mlPred/',help="output path for plots and tables")
 
    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False, help="disable X-term for batch mode")

    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--dataName", type=str, default='210611_3_NI-a0.17', help="experimental data ")
    args = parser.parse_args()
    args.prjName='neurInfer'
    args.formatVenue='prod'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.isdir(args.outPath):  os.makedirs(args.outPath)
    return args


#...!...!..................
def model_infer_exper(model,loader):
    device=torch.cuda.current_device()   
    model.eval()
    criterion =torch.nn.MSELoss().to(device) # Mean Squared Loss
    print('loader size=',len(loader))
    assert len(loader)==1
    with torch.no_grad():
        waves, target  = next(iter(loader))
        waves_dev, target_dev = waves.to(device), target.to(device)
        output_dev = model(waves_dev)
        lossOp=criterion(output_dev, target_dev)
        #print('qq',lossOp,len(loader.dataset),len(loader))
        loss = lossOp.item()
        output=output_dev.cpu()
        
    print('infere done, nSample=%d  loss=%.4f'%(target.shape[0],loss),flush=True)
    return np.array(waves), np.array(target) , np.array(output), float(loss)


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
  args=get_parser()
  logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

  sumF=args.modelPath+'/sum_train.yaml'
  trainMD = read_yaml( sumF)
  parMD=trainMD['train_params']
  inpMD=trainMD['input_meta']
  
  assert torch.cuda.is_available() 
  model=load_model(trainMD,args.modelPath)
  #1print(model)

  # ... prime data-loader
  parMD['cell_name']=args.dataName  
  parMD['world_size']=1
  domain='exper'
  parMD['data_path']=args.dataPath
  parMD['shuffle']=False # to assure sync with other data records
  inpMD['h5nameTemplate']='*.h5'  
  parMD['train_conf']['recover_upar_from_ustar']=False    
  loader = get_data_loader(parMD,  inpMD,domain, verb=1)

  if 1:  # hack: read all data again to access meta-data
      print('M: re-read data for auxiliary info:')
      
      inpF=parMD['full_h5name']
      bigD,expMD=read3_data_hdf5(inpF)
      print('M:expMD:',expMD)
      
  waves, U,Z, loss =model_infer_exper(model,loader)
  
  sumRec={}
  sumRec['domain']=domain

  sumRec['numSamples']=float(U.shape[0])
  sumRec['short_name']=args.dataName+'_'+str(trainMD['job_id'])
  #sumRec['train_info']= trainMD
  sumRec['exper_info']= expMD

  bigD['pred_upar']=Z
  bigD['true_upar']=U
  bigD['waves']=waves
  
  outF=sumRec['short_name']+'.mlPred.h5'
  write3_data_hdf5(bigD,args.outPath+outF,metaD=sumRec,verb=1)

  #print('DL-conf:'); pprint(loader.dataset.conf)
  print('predZ:',Z,flush=True)
  #
  #  - - - -  only plotting code is below - - - - -
  
  plot=Plotter_NeuronInverter(args,inpMD ,sumRec )
  
  plot.params1D(Z,'pred Z',figId=8,doRange=False)

  if 'exp4ml' in args.dataPath:
      plot.params_vs_expTime(Z,bigD, figId=9)

  if 0: 
    print('input data example, it will plot waveforms')
    dlit=iter(data_loader)
    xx, yy = next(dlit)
    #1xx, yy = next(dlit) #another sample
    print('batch, X,Y;',xx.shape,xx.dtype,yy.shape,yy.dtype)
    print('Y[:2]',yy[:2])
    plot.frames_vsTime(xx,yy,9)
   
  
  plot.display_all('exper')  

