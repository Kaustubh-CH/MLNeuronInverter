#!/usr/bin/env python3
""" 
format samples for ML training

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os,sys
from toolbox.Util_H5io3 import   read3_data_hdf5, write3_data_hdf5,append_data_hdf5
from pprint import pprint
import numpy as np
#from vet_volts import tag_zeros
from aggregate_Kaustubh import normalize_volts
import h5py,json
import argparse
import time
import csv

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataPath",help="input/output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3_march06')
    parser.add_argument("--cellName", type=str, default='ALL_CELLS', help="cell name list, blanks separated")
    parser.add_argument("--conf", type=int, default=1, help="output configuration")
    parser.add_argument("--addStim", type=bool, default=False, help="output configuration")
    
    args = parser.parse_args()
    args.verb=1
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

def get_normal_stim():
    
    #Read stims and normalize
    norm_stim =[]
    stimPath='/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/'
    for stims in stimNL:
        inpF=os.path.join(stimPath,stims)
        inpF+=".csv"
        file = open(inpF,"r")
        data = list(csv.reader(file, delimiter=","))
        file.close()
        data = [float(row[0]) for row in data]
        data = np.array(data[1000:])
        #normalizing between 0 to 1
        # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        norm_stim.append(data)
        # plt.plot(data)
    norm_stim = (norm_stim-np.min(norm_stim))/(np.max(norm_stim)-np.min(norm_stim))
    
    # import matplotlib.pyplot as plt
    # for i in range(norm_stim.shape[0]):
    #     data = norm_stim[i]
    #     plt.plot(data)

    # plt.savefig("Data_norm2.png")
    return np.transpose(norm_stim)




#...!...!..................
def format_raw(dom,off_len,addStim=False):
    [ioff,myLen]=off_len
    print('\nformat:', dom,[ioff,myLen])

    for xN in simD:
        yN='%s_%s'%(dom,xN)
        bigD[yN]=simD[xN]
        # bigD[yN]=simD[xN][ioff:ioff+myLen]

        #print('new ',dom,yN,bigD[yN].shape)

    # .... normalize and QA volts ...
    volts=bigD.pop(dom+'_volts')
    print('normalize and QA:',volts.shape,dom)
    assert args.conf==1 # change here for different packing & normalization schemes
    volts_norm,nFlat=normalize_volts(volts,dom, perProbe=False)
    if(addStim):
        stim_norm = get_normal_stim()
        stim_norm = stim_norm.reshape(volts_norm.shape[1],1,volts_norm.shape[3])
        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(stim_norm.shape[2]): 
        #     data = np.squeeze(stim_norm[:,:,i])
        #     plt.plot(data)
        # plt.savefig("Data_norm3.png")

        stim_norm = np.repeat(stim_norm[np.newaxis, :, :, :], volts_norm.shape[0], axis=0)
        volts_norm = np.concatenate([stim_norm, volts_norm], axis=2)
    


    bigD[dom+'_volts_norm']=volts_norm
    return int(nFlat)

def write_meta_json_hdf5(bigD,outF,metaD):
    h5f = h5py.File(outF, 'w')
    if metaD!=None:
        metaJ=json.dumps(metaD)
        # if type(metaJ)!=np.ndarray: # packs a single value in ot np-array
        #     metaJ=np.array([metaJ])
        # h5f.create_dataset('meta.JSON', data=metaJ)
        dtvs = h5py.special_dtype(vlen=str)
        dset = h5f.create_dataset('meta.JSON', (1,), dtype=dtvs)
        
        dset[0]=metaJ
        print('h5-write :','meta.JSON', 'as string',dset.shape,dset.dtype)
    #Creating just the shell for 
    for items in bigD:
        shape_thread = bigD[items].shape
        shape_all=list(shape_thread)
        if('train' in items):
            shape_all[0]=train_val
        else:
            shape_all[0]=nval
        dataset = h5f.create_dataset(items, shape=shape_all, dtype=bigD[items].dtype)
    h5f.close()

def append_data_hdf5_index(bigD,outF,metaD,thread_id=0,thread_total=1):
    h5f = h5py.File(outF, 'a')
    for items in bigD:
        rec=bigD[items]
        if type(rec)!=np.ndarray: # packs a single value in ot np-array
            rec=np.array([rec])
        dataset = h5f[items]
        index=dataset.shape[0]
        len_thread=int(index/thread_total)
        #HDF5 loads that particular slice instead of the entire dataset.
        dataset[thread_id*len_thread:(thread_id+1)*len_thread]=bigD[items]
    h5f.close()




def read_meta_json_hdf5(inpF):
    h5f = h5py.File(inpF, 'r')
    try:
        inpMD=json.loads(h5f['meta.JSON'][0])
    except:
        inpMD=None
    h5f.close()
    return inpMD

def read3_only_data_hdf5(inpF,verb=1,skipKey=None,ind1=0,ind2=-1):
   
    if verb>0:
            print('read hdf5 from :',inpF)
            if skipKey!=None:  print('   h5 skipKey:',skipKey)
            # start = time.time()
    h5f = h5py.File(inpF, 'r')
    objD={}
    for x in h5f.keys():
        if verb>1: print('item=',x,type(h5f[x]),h5f[x].shape,h5f[x].dtype)
        if skipKey!=None:
            skip=False            
            for y in skipKey:
                if y in x: skip=True
            if skip: continue
        if h5f[x].dtype==object:
            obj=h5f[x][0]
            #print('bbb',type(obj),obj.dtype)
            if verb>0: print('read str:',x,len(obj),type(obj))
        else:
            obj=h5f[x][ind1:ind1+ind2]
            if verb>0: print('read obj:',x,obj.shape,obj.dtype)
        objD[x]=obj
    try:
        objD.pop('meta.JSON')
    except:
        print('No MetaData')
    h5f.close()
    return objD
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__=="__main__":
  args=get_parser()
    # thread_id=int(os.environ['SLURM_PROCID'])
    # thread_total = int(os.environ['SLURM_NTASKS'])
  thread_total=50
  for thread_id in range(0,thread_total):  
    inpF0=args.cellName+'.simRaw.h5'
    inpF=os.path.join(args.dataPath,inpF0)
    # simD,simMD=read3_data_hdf5(inpF)
    simMD=read_meta_json_hdf5(inpF)
    if 0:
        pprint(simMD); exit(9)
    
    #... isolate stims to be not split
    stimNL=simMD['simu_info']['stim_names']
    
    
    # ... split data into domains
    totSamp=simMD['simu_info']['num_total_samples']
    nval=int(totSamp/10)
    tval =int(nval/thread_total)
    train_val=totSamp-2*nval #
    thread_train_val=int(train_val/thread_total)
    per_thread_totSamp = int(totSamp/thread_total)


    # compute offest and length per domain  (Offset,Length)
    split_index={'valid':[0+thread_id*tval,tval], 'test':[nval+thread_id*tval,tval],'train':[2*nval+thread_id*thread_train_val, thread_train_val]}
    split_index_total = {'valid':[0,nval],'test':[nval,nval],'train':[2*nval,train_val]}
    split_index={'valid':[0+thread_id*per_thread_totSamp,tval], 'test':[tval+thread_id*per_thread_totSamp,tval],'train':[2*tval+thread_id*per_thread_totSamp, thread_train_val]}
    print('M:split_index',split_index)
    print("M:Thread_id",thread_id)
    bigD={}  # this will be output
    totFlat=0
    
    # keep this order to deal with bigest record first - it may not matter
    for dom in ['train','valid','test']: 
        simD=read3_only_data_hdf5(inpF,1,None,split_index[dom][0],split_index[dom][1])
        #... isolate stims to be not split
        tmpD={xN:simD.pop(xN) for xN in stimNL}
        
        totFlat+=format_raw(dom,split_index[dom],args.addStim)
        
    #.... update meta data
    pmd={'split_index':split_index_total,'num_flat_volts':totFlat,'pack_conf':args.conf,'full_input_h5':inpF0}
    simMD['pack_info']=pmd
    if(args.addStim):
        probe = ['stim']
        probe = probe + simMD['simu_info']['probe_names']
        simMD['simu_info']['probe_names'] = probe
    #... add stims back
    
   
    #print('M:sim meta-data');   pprint(simMD)
    #print('M:big',list(bigD))

    print('M:pack_info');pprint(pmd)
    
    outF0='%s.mlPack%d.h5'%(args.cellName,args.conf)
    outF=os.path.join(args.dataPath,outF0)
    if(thread_id==0): #SAVING data is left
        write_meta_json_hdf5(bigD,outF,simMD)

    #time.sleep(30)
    append_data_hdf5_index(bigD,outF,metaD=simMD,thread_id=thread_id,thread_total=thread_total)
    #Add stims seperately 
    for xN in stimNL: bigD[xN]=tmpD[xN]
    if(thread_id==0):
        append_data_hdf5(tmpD,outF,metaD=None,verb=1)
    print('M:done')

    if 1:
        from toolbox.Util_IOfunc import write_yaml # for testing only
        write_yaml(simMD,'aa.yaml')
