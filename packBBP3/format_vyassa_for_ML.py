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
from aggregate_Kaustubh import normalize_volts
from toolbox.Util_IOfunc import write_yaml, read_yaml

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--outPath",help="output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3_march06')
    parser.add_argument("--inpPath",help="input  path",  default='/global/cfs/cdirs/m2043/balewski/neuronBBP-pack8kHzRam/probe_4pr8kHz')
    parser.add_argument("--vyassaName", type=str, default='bbp153', help="cell name: bbpNNN")
    parser.add_argument("--conf", type=int, default=1, help="output configuration")
    
    args = parser.parse_args()
    args.verb=1
    args.inpTag='cellSpike_4pr8kHz'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def rebuildMD(inpD):
    dai=inpD['dataInfo']
    rai=inpD['rawInfo']
    md={}
        
    pn=[]; pr=[]; pkeep=[]
    for i,rec in enumerate(dai['physRange']):
        #print('pp',i,rec)
        name,lb,ub=rec
        if 'fixed' in name: continue
        assert name==dai['parName'][i]
        base=float(np.sqrt(lb*ub))
        pn.append(name)
        pr.append([base,1.0, 'S/cm2'])
        pkeep.append(i)
    md['phys_par_range']=pr
    md['parName']= pn
    

    md['cell_name']=rai['bbpName']
    md['linearParIdx']=[]
    md['num_phys_par']=len(pr)
    md['num_time_bins']=dai['numTimeBin']
    md['stimParName']=['stim_mult', 'stim_offset']
    md['stimParRange']=[[1, 0.3e-10], [0, 0.3e-10]]
    md['timeAxis']=rai['timeAxis']

    si={}
    si['NEURON_ver']='vyassa-neuron-2020'
    si['origData']=inpF
    si['bbp_name_clone']=rai[ 'rawDataName'].split('-')[0]
    si['full_probe_names']=dai['featureName']
    si['probe_names']=si['full_probe_names']  # just use the same
    si['jobId']=rai['rawJobIdL']
    si['num_probs']=len(si['probe_names'])
    si['stim_names']=[rai['stimName']]
    si['num_stims']=1
    si['num_total_samples']=dai['totalFrameCount']
    
    
    md['simu_info']=si

    return md,pkeep
    
#...!...!..................
def addStim():
    stimN=simMD['simu_info']['stim_names'][0]
    inpF=inpPath+'stim.%s.yaml'%stimN
    stimD=read_yaml(inpF)
    pprint(stimD)
    bigD[stimN]=stimD['stimFunc']
    
#...!...!..................
def format_raw(dom,inpD):    
    for xN in ['phys_par','unit_par']:
        yN='%s_%s'%(dom,xN)
        bigD[yN]=inpD[xN][:,pkeep]  # skip fixed params
        print('add ',dom,yN,bigD[yN].shape)

    volts=inpD['frames']
    shp=volts.shape
    volts=volts.reshape(shp+(1,))  # account for stim dimension, here just 1
    print('normalize and QA:',volts.shape,dom)
    assert args.conf==1 # change here for different packing & normalization schemes
    volts_norm,nFlat=normalize_volts(volts,dom, perProbe=False)
    bigD[dom+'_volts_norm']=volts_norm
    print('nFlat:',nFlat)
    return int(nFlat)
                 
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    if 0:  # test example output
        inpF='/global/cfs/cdirs/m2043/balewski/neuronBBP3-10kHz_3pr_6stim/jan12Nrow_mlPack1/L5_TTPC1cADpyr0.mlPack1.h5'
        simD,mm=read3_data_hdf5(inpF)
        pprint(mm)
        ok99
    
    inpPath=os.path.join(args.inpPath,args.vyassaName)+'/'
    inpF=inpPath+'meta.%s.yaml'%args.inpTag
    inpMD=read_yaml(inpF)
    pprint(inpMD)
    
    simMD,pkeep=rebuildMD(inpMD)
    print('\nM:======')
    npar=len(pkeep)
    print('keep %d params:'%npar,pkeep)
    #print(simMD['parName'])
    #pprint(simMD)
    
    #.... repack HD5
    bigD={}
    addStim()

    totFlat=0
    for dom in ['test','val','train']:        
        inpF=inpPath+'%s.%s.data_%s.h5'%(args.vyassaName,args.inpTag,dom)
        print('read dom=',inpF)
        simD,_=read3_data_hdf5(inpF)
        if dom=='val': dom='valid'
        totFlat+=format_raw(dom,simD)
        #break #tmp
       
    #.... update meta data
    pmd={'num_flat_volts':totFlat,'pack_conf':args.conf,'full_input_h5':None}
    pmd['split_index']=inpMD['dataInfo']['splitIdx']
    simMD['pack_info']=pmd
   
    #print('M:sim meta-data');   pprint(simMD)
    #print('M:big',list(bigD))

    print('M:pack_info');pprint(pmd)
    
    outF0='%s.mlPack%d.h5'%(args.vyassaName,args.conf)
    outF=os.path.join(args.outPath,outF0)
    write3_data_hdf5(bigD,outF,metaD=simMD)
    print('M:done')

    if 1:
        from toolbox.Util_IOfunc import write_yaml # for testing only
        write_yaml(simMD,'aa.yaml')
