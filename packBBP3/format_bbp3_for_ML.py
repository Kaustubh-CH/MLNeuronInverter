#!/usr/bin/env python3
""" 
format samples for ML training

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os,sys
from toolbox.Util_H5io3 import   read3_data_hdf5, write3_data_hdf5
from pprint import pprint
import numpy as np
#from vet_volts import tag_zeros
from aggregate_Kaustubh import normalize_volts

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataPath",help="input/output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3_dec26')
    parser.add_argument("--cellName", type=str, default='L4_SScADpyr4', help="cell name list, blanks separated")
    parser.add_argument("--conf", type=int, default=1, help="output configuration")
    
    args = parser.parse_args()
    args.verb=1
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!..................
def format_raw(dom,off_len):
    [ioff,myLen]=off_len
    print('format:', dom,[ioff,myLen])

    for xN in simD:
        yN='%s_%s'%(dom,xN)
        bigD[yN]=simD[xN][ioff:ioff+myLen]
        #print('new ',dom,yN,bigD[yN].shape)

    # .... normalize and QA volts ...
    volts=bigD.pop(dom+'_volts')
    print('normalize and QA:',volts.shape,dom)
    assert args.conf==1 # change here for different packing & normalization schemes
    volts_norm,nFlat=normalize_volts(volts,dom)
    bigD[dom+'_volts_norm']=volts_norm
    return int(nFlat)
                 
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    
    inpF0=args.cellName+'.simRaw.h5'
    inpF=os.path.join(args.dataPath,inpF0)
    simD,simMD=read3_data_hdf5(inpF)

    #... isolate stims to be not split
    stimNL=simMD['simu_info']['stim_names']
    tmpD={xN:simD.pop(xN) for xN in stimNL}
    
    # ... split data into domains
    totSamp=simMD['simu_info']['num_total_samples']
    nval=int(totSamp/10)
    # compute offest and length per domain
    split_index={'valid':[0,nval], 'test':[nval,2*nval],'train':[2*nval, totSamp-2*nval]}
    print('M:split_index',split_index)
    bigD={}  # this will be output
    totFlat=0
    for dom in split_index:
        totFlat+=format_raw(dom,split_index[dom])
        
    #.... update meta data
    pmd={'split_index':split_index,'num_flat_volts':totFlat,'pack_conf':args.conf,'full_input_h5':inpF0}
    simMD['pack_info']=pmd

    #... add stims back
    for xN in stimNL: bigD[xN]=tmpD[xN]
   
    #print('M:sim meta-data');   pprint(simMD)
    #print('M:big',list(bigD))

    print('M:pack_info');pprint(pmd)
    
    outF0='%s.mlPack%d.h5'%(args.cellName,args.conf)
    outF=os.path.join(args.dataPath,outF0)
    write3_data_hdf5(bigD,outF,metaD=simMD)
    print('M:done')

    if 1:
        from toolbox.Util_IOfunc import write_yaml # for testing only
        write_yaml(simMD,'aa.yaml')
