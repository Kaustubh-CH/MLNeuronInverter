#!/usr/bin/env python3
""" 
read trained net : model+weights
read test data from HD5
infere for  test data 

Inference works alwasy on 1 GPU or CPUs

 ./predict.py  --modelPath /global/cfs/cdirs/mpccc/balewski/tmp_neurInv/exp_excite/70257/out


"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import torch

import  time
import sys,os
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Plotter import Plotter_NeuronInverter
from toolbox.Dataloader_H5 import get_data_loader
from pprint import pprint
import argparse

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--facility", default='corigpu', type=str)
    parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")

    parser.add_argument("-m","--modelPath",  default='/global/cscratch1/sd/balewski/tmp_digitalMind/neuInv/manual/', help="trained model ")
    parser.add_argument("--dom",default='test', help="domain is the dataset for which predictions are made, typically: test")

    parser.add_argument("-o", "--outPath", default='same',help="output path for plots and tables")
 
    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False, help="disable X-term for batch mode")

    parser.add_argument("-n", "--numSamples", type=int, default=None, help="limit samples to predict")
    parser.add_argument("--stimsSelect",default=None, type=int, nargs='+', help="list of stims, space separated")
    parser.add_argument("--testStimsSelect",default=None, type=int, nargs='+', help="list of stims, space separated")
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='ab', nargs='+',help="abcd-string listing shown plots")

    parser.add_argument("--cellName", type=str, default=None, help="alternative data file name ")
    parser.add_argument("--idx", nargs='*' ,required=False,type=str, default=None, help="Included parameters")
    args = parser.parse_args()
    args.prjName='nif'
    args.outPath+'/'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def load_model(trainMD,modelPath):
    # ... assemble model

    device = torch.device("cuda")
    # load entirel model
    modelF = os.path.join(modelPath, trainMD['train_params']['blank_model'])
    stateF= os.path.join(modelPath, trainMD['train_params']['checkpoint_name'])

    model = torch.load(modelF)
    model2 = torch.nn.DataParallel(model)
    allD=torch.load(stateF, map_location=str(device))
    print('all model ok',list(allD.keys()))
    stateD=allD["model_state"]
    keyL=list(stateD.keys())
    if 'module' not in keyL[0]:
      ccc={ 'module.%s'%k:stateD[k]  for k in stateD}
      stateD=ccc
    model2.load_state_dict(stateD)
    return model2

#...!...!..................
def model_infer(model,test_loader,trainMD):
    device=torch.cuda.current_device()   

    model.eval()
    criterion =torch.nn.MSELoss().to(device) # Mean Squared Loss
    test_loss = 0

    # prepare output container, Thorsten's idea
    num_samp=len(test_loader.dataset)
    outputSize=trainMD['train_params']['model']['outputSize']
    print('predict for num_samp=',num_samp,', outputSize=',outputSize)
    # clever list-->numpy conversion, Thorsten's idea
    Uall=np.zeros([num_samp,outputSize],dtype=np.float32)
    Zall=np.zeros([num_samp,outputSize],dtype=np.float32)
    nEve=0
    nStep=0
    with torch.no_grad():
        for data, target in test_loader:
            data_dev, target_dev = data.to(device), target.to(device)
            output_dev = model(data_dev)
            lossOp=criterion(output_dev, target_dev)
            #print('qq',lossOp,len(test_loader.dataset),len(test_loader)); ok55
            test_loss += lossOp.item()
            output=output_dev.cpu()
            nEve2=nEve+target.shape[0]
            #print('nn',nEve,nEve2)
            Uall[nEve:nEve2,:]=target[:]
            Zall[nEve:nEve2,:]=output[:]
            nEve=nEve2
            nStep+=1
    test_loss /= nStep
    print('infere done, nEve=%d nStep=%d loss=%.4f'%(nEve,nStep,test_loss),flush=True)
    return test_loss,Uall,Zall


#...!...!..................
def compute_residual(trueU,recoU,md,idx):
    parName=md['parName']
    # idx=[0,1,2,3,4,5,6,7,8,9,10,13,14,15]
    parName=[parName[id] for id in idx]
    nPar=len(parName)
    outL=[]
    for iPar in range(0,nPar):
        D=trueU[:,iPar] - recoU[:,iPar]
        resM=D.mean()
        resS=D.std()
        outL.append( [ parName[iPar], float(resM),float(resS) ] )
        print('#res,%d,%.4f'%(iPar,resS))
        #print('#res,%d,%s'%(iPar,parName[iPar]))
    #pprint(outL)
    return outL

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
  args=get_parser()

  if args.outPath=='same' : args.outPath=args.modelPath
  sumF=args.modelPath+'/sum_train.yaml'
  trainMD = read_yaml( sumF)
  parMD=trainMD['train_params']
  inpMD=trainMD['input_meta']
  #pprint(inpMD)
  assert torch.cuda.is_available() 
  model=load_model(trainMD,args.modelPath)
  #1print(model)

  if args.cellName!=None:
      parMD['cell_name']=args.cellName
      
  if args.numSamples!=None:
      parMD['data_conf']['max_glob_samples_per_epoch' ] = args.numSamples

  if args.testStimsSelect!=None:
      parMD['data_conf']['valid_stims_select' ] = args.testStimsSelect

  if args.stimsSelect!=None:
    #   assert  parMD['data_conf']['serialize_stims']==True 
      parMD['data_conf']['stims_select' ] = args.stimsSelect
      args.prjName='nistim'+''.join(['%d'%i for i in args.stimsSelect] )
      print('M: prjName',args.prjName)
    
  domain=args.dom
  idx=range(len(inpMD['parName']))
  if(args.idx is not None):
      idx= args.idx
      idx =[int(i) for i in idx]
  parMD['world_size']=1
  #pprint(parMD); ok6
  
  data_loader = get_data_loader(parMD, domain, verb=1)

  startT=time.time()
  loss,trueU,recoU=model_infer(model,data_loader,trainMD)
  predTime=time.time()-startT
  print('M: infer : Average loss: %.4f  dom=%s samples=%d , elaT=%.2f min\n'% (loss, domain, trueU.shape[0],predTime/60.))

  residualL=compute_residual(trueU,recoU,inpMD,idx)
  print('#res,job,%s'%trainMD['job_id'])
  print('#res,MSEloss,%.4f'%loss)
  print('#res,samples,%d\n'%trueU.shape[0])
    
  sumRec={}
  sumRec['domain']=domain
  sumRec['jobId']=trainMD['job_id']
  sumRec[domain+'LossMSE']=float(loss)
  sumRec['predTime']=predTime
  sumRec['numSamples']=trueU.shape[0]
  sumRec['lossThrHi']=0.40  # for tagging plots
  sumRec['inpShape']=trainMD['train_params']['model']['inputShape']
  sumRec['short_name']=trainMD['train_params']['cell_name']
  sumRec['modelDesign']=trainMD['train_params']['model']['myId']
  sumRec['trainTime']=trainMD['trainTime_sec']
  sumRec['loss_valid']= trainMD['loss_valid']
  sumRec['train_stims_select']= trainMD['train_stims_select']
  sumRec['train_glob_sampl']= trainMD['train_glob_sampl']
  sumRec['pred_stims_select']= trainMD['train_params']['data_conf']['stims_select']
  sumRec['residual_mean_std']=residualL

  outN='sum_pred_%s.yaml'%args.prjName
  write_yaml(sumRec, os.path.join(args.outPath,outN))
    
  #  - - - -  only plotting code is below - - - - -  
  plot=Plotter_NeuronInverter(args,inpMD ,sumRec )
  
  if 'a' in args.showPlots:
      plot.param_residua2D(trueU,recoU)
  if 'b' in args.showPlots:
      plot.params1D(recoU,'pred U',figId=8)
  if 'c' in args.showPlots:
      plot.params1D(trueU,'true U',figId=7)

  if 0: 
      print('input data example, it will plot waveforms')
      dlit=iter(data_loader)
      xx, yy = next(dlit)
      #1xx, yy = next(dlit) #another sample
      print('batch, X,Y;',xx.shape,xx.dtype,yy.shape,yy.dtype)
      print('Y[:2]',yy[:2])
      plot.frames_vsTime(xx,yy,9)
   
  figN=domain+'_'+ parMD['cell_name']

  plot.display_all(figN, png=1)  

