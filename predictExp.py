#!/usr/bin/env python3
""" 
PREDiction Kaustubh


"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import torch
import pandas as pd

import  time
import sys,os
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from toolbox.Util_H5io3 import write3_data_hdf5

from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Plotter import Plotter_NeuronInverter
from toolbox.Dataloader_H5 import get_data_loader
from pprint import pprint
import argparse
param_names = ['gNaTs2_tbar_NaTs2_t_apical',
'gSKv3_1bar_SKv3_1_apical',
'gImbar_Im_apical',
'gIhbar_Ih_dend',
'gNaTa_tbar_NaTa_t_axonal',
'gK_Tstbar_K_Tst_axonal',
'gNap_Et2bar_Nap_Et2_axonal',
'gSK_E2bar_SK_E2_axonal',
'gCa_HVAbar_Ca_HVA_axonal',
'gK_Pstbar_K_Pst_axonal',
'gCa_LVAstbar_Ca_LVAst_axonal',
'g_pas_axonal',
'cm_axonal',
'gSKv3_1bar_SKv3_1_somatic',
'gNaTs2_tbar_NaTs2_t_somatic',
'gCa_LVAstbar_Ca_LVAst_somatic',
'g_pas_somatic',
'cm_somatic',
'e_pas_all',]
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
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--expFile",  default='/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/', help="Experimental data path")
    
    parser.add_argument("--saveFile",  default='./unitParams', help="Experimental data path")
    parser.add_argument("--cellName", type=str, default=None, help="alternative data file name ")
    parser.add_argument("--predictStim",default=0 , type=int, nargs='+', help="list of stims, space separated")
    parser.add_argument("--idx", nargs='*' ,required=False,type=str, default=None, help="Included parameters")
    parser.add_argument("--test-plot-size",default=None , type=int, help="size of test plot")
    
    
    args = parser.parse_args()
    args.prjName='neurInfer'
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
    sweepId=65
    # os.mkdir("./unitParams")
    try:
      os.mkdir(trainMD['saveFile'])
    except:
      print("Directory already exists",trainMD['saveFile'])
  
    with torch.no_grad():
        for data, target in test_loader:
            sweepId+=1
            data_dev, target_dev = data.to(device), target.to(device)
            output_dev = model(data_dev)
            print(output_dev.size())

            dataframe={"unit_params_predict":output_dev.squeeze().tolist(),"unit_params_actual":target_dev.squeeze().tolist(),"param_names":param_names}
            df = pd.DataFrame(dataframe)
            
            df.to_csv(trainMD['saveFile']+"/unitParam"+str(sweepId)+".csv",index=False)
            # if(trainMD['test_plot_size']!=None)
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

def compute_dummy_residual(md,idx):
    parName=md['parName']
    # idx=[0,1,2,3,4,5,6,7,8,9,10,13,14,15]
    parName=[parName[id] for id in idx]
    nPar=len(parName)
    outL=[]
    for iPar in range(0,nPar):
        outL.append( [ parName[iPar], float(0.1),float(0.1) ] )
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
  param_names=trainMD['input_meta']['parName']
  model=load_model(trainMD,args.modelPath)
  #1print(model)
  idx=range(len(inpMD['parName']))
  # if(args.idx is not None):
  #     idx= args.idx
  #     idx =[int(i) for i in idx]
  if('include' in inpMD.keys()):
        idx=inpMD['include']
  else:
        idx = range(len(inpMD['parName']))

  param_names=[param_names[id] for id in idx]

  if args.cellName!=None:
      parMD['cell_name']=args.cellName
      
  if args.numSamples!=None:
      parMD['max_local_samples_per_epoch' ] = args.numSamples
  domain=args.dom

  parMD['world_size']=1
  trainMD['test_plot_size']=None
  parMD['local_batch_size']=1
  if(args.test_plot_size!=None):
      parMD['data_conf']['max_glob_samples_per_epoch']=args.test_plot_size
      trainMD['test_plot_size']=args.test_plot_size
    
  else:
    parMD['data_path']='/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/L5_TTPC1cADpyr2.mlPack1.h5'
    parMD['data_conf']['data_path']='/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/'
    # parMD['data_path_temp']='/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/'
    parMD['data_path_temp'] = args.expFile
    parMD['cell_name']='L5_TTPC1cADpyr2'
    parMD['data_conf']['probs_select']=[0]
    parMD['data_conf']['stims_select']=args.predictStim
    parMD['data_conf']['valid_stims_select']=args.predictStim
    parMD['exp_data']=True
  parMD['shuffle']=False
  trainMD['saveFile']=args.saveFile

  data_loader = get_data_loader(parMD, domain, verb=1)

  startT=time.time()
  loss,U,Z=model_infer(model,data_loader,trainMD)
  predTime=time.time()-startT
  print('M: infer : Average loss: %.4f  dom=%s samples=%d , elaT=%.2f min\n'% (loss, domain, Z.shape[0],predTime/60.))
  if(len(data_loader.dataset)<30):

    simulated_data  =np.array([batch.numpy() for batch,_ in data_loader])
    simulated_data = simulated_data.squeeze()
    simulated_data_hdf={}
    simulated_data_hdf['volts']=simulated_data
    simulated_data_hdf_meta={'cell':parMD['cell_name'],'dom':'TestPlot'}
    
    write3_data_hdf5(simulated_data_hdf,args.saveFile+'inputVolts.hdf5',simulated_data_hdf_meta)
  else:
    print("Not saving h5py because size>30",len(data_loader.dataset))
  

  residualL = compute_dummy_residual(inpMD,idx)
  sumRec={}
  sumRec['domain']=domain
  sumRec['jobId']=trainMD['job_id']
  sumRec[domain+'LossMSE']=float(loss)
  sumRec['predTime']=predTime
  sumRec['numSamples']=U.shape[0]
  sumRec['lossThrHi']=0.40  # for tagging plots
  sumRec['inpShape']=trainMD['train_params']['model']['inputShape']
  sumRec['short_name']=trainMD['train_params']['cell_name']
  sumRec['modelDesign']=trainMD['train_params']['model']['myId']
  sumRec['trainRanks']=trainMD['train_params']['world_size']
  sumRec['trainTime']=trainMD['trainTime_sec']
  sumRec['loss_valid']= trainMD['loss_valid']
  sumRec['train_stims_select']= trainMD['train_stims_select']
  sumRec['train_glob_sampl']= trainMD['train_glob_sampl']
  sumRec['pred_stims_select']= trainMD['train_params']['data_conf']['stims_select']
 
  sumRec['residual_mean_std']=residualL

  #  - - - -  only plotting code is below - - - - -
  if(args.test_plot_size!=None):
    plot=Plotter_NeuronInverter(args,inpMD ,sumRec )
    plot.param_residua2D(U,Z)

  write_yaml(sumRec, args.outPath+'/sum_predExp.yaml')

  #1plot.params1D(U,'true U',figId=7)
#   plot.params1D(Z,'pred Z',figId=8)

  if 0: 
    print('input data example, it will plot waveforms')
    dlit=iter(data_loader)
    xx, yy = next(dlit)
    #1xx, yy = next(dlit) #another sample
    print('batch, X,Y;',xx.shape,xx.dtype,yy.shape,yy.dtype)
    print('Y[:2]',yy[:2])
    # plot.frames_vsTime(xx,yy,9)
   
    if(args.test_plot_size!=None):
      figN=domain+'_'+ parMD['cell_name']

      plot.display_all(figN, png=1)  

