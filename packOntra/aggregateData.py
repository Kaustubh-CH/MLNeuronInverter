#!/usr/bin/env python
""" 
uses multi-h5 data reader to repack multiple files into one monster file
take exactly the same number of samples from each finput file
Dependendy: needs 1 train summary file to get al paths

Naming convention for  out files: 
Excitatory use 100% of samples: practice10c (144G ),  witness2c (29G)
Excitatory+clones : practice50c 29M samp (859G ),  witness13c 7.5M samp (224G)
Inhibitory use 30% of samples: practice140c (472G) witness17c(58G)  october12c(130G)

Do it on big memory node

ssh cori
module load cmem  pytorch
salloc -C amd -q bigmem -t 2:00:00


"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np


import  time
import sys,os,json
from pprint import pprint
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
    
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Util_H5io3 import write3_data_hdf5
from toolbox.Dataloader_multiH5 import get_data_loader
import argparse

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outPath", default='/global/cscratch1/sd/balewski/out2/',help="output path for data")
    parser.add_argument("--outName", default='repack1',help="output data name")
    parser.add_argument("--cellName", type=str, default=['bbp153','bbp102'],nargs="+", help="cell shortName, can be  list ")
    parser.add_argument("-n", "--numSamplesPerFile", type=int, default=2000, help="samples to read")
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
    args = parser.parse_args()
    args.prjName='neurInfer'
    args.formatVenue='prod'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    sumF='out/sum_train.yaml'
    sumMD = read_yaml( sumF)
    parMD=sumMD['train_params']
    inpMD=sumMD['input_meta']
    parMD['world_size']=1
    parMD['world_rank']=0
    parMD['shuffle']=True
    if 'bbp' in args.cellName[0]:
        parMD['cell_name']=args.cellName
    else:  # practice or witness : use all
        parMD['cell_name']=inpMD['cellSplit'][args.cellName[0]]
    parMD['local_batch_size']=222

    pprint(parMD)
    #pprint(inpMD)

    numCell=len(parMD['cell_name'])
    bigData={}
    startT=time.time()
    for dom in [ 'train','test', 'val']:
        
        parMD['numLocalSamples']=args.numSamplesPerFile*numCell
        if dom!='train': parMD['numLocalSamples']//=8
        ds = get_data_loader(parMD,  inpMD,dom, verb=1, onlyDataset=True)
        print('ingested dom=%s, size=%d'%(dom,len(ds)),ds.data_frames.dtype,ds.data_frames.shape)
        bigData[dom+'_frames']=ds.data_frames
        bigData[dom+'_unitStar_par']=ds.data_parU
                
    predTime=time.time()-startT
    print('M: train=%s repack elaT=%.2f min\n'% (  str(bigData['train_frames'].shape),predTime/60.))
    outF=inpMD['h5nameTemplate'].replace('*',args.outName)
    outMD={'cellNameList':args.cellName, 'globalTrainSamples':args.numSamplesPerFile*numCell}
    outMD['h5name']=outF
    outMD['numInpFiles']=len(args.cellName)
    print('metaD:',outMD)

    write3_data_hdf5(bigData,args.outPath+'/'+outF,metaD=outMD)
    print('M: done')
