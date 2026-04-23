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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import efel
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

EFEL_FEATURE_NAMES = [
    'mean_frequency', 'AP_amplitude', 'AHP_depth_abs_slow',
    'fast_AHP_change', 'AHP_slow_time',
    'spike_half_width', 'time_to_first_spike', 'inv_first_ISI', 'ISI_CV',
    'ISI_values', 'adaptation_index'
]

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataPath",help="input/output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3_march06')
    parser.add_argument("--cellName", type=str, default='ALL_CELLS', help="cell name list, blanks separated")
    parser.add_argument("--conf", type=int, default=1, help="output configuration")
    parser.add_argument("--addStim", type=bool, default=False, help="output configuration")
    parser.add_argument("--thread_total", type=int, default=10, help="number of threads")
    
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


def _process_efel_chunk(chunk, time_array, stim_start, stim_end, chunk_idx):
    """
    Helper function to process a single chunk of traces for EFEL feature extraction.
    Returns (chunk_idx, features_array) for reassembly in original order.
    """
    n_features = len(EFEL_FEATURE_NAMES) + 1
    chunk_out = np.zeros((chunk.shape[0], n_features), dtype=np.float32)
    
    trace_data_list = [
        {
            'T': time_array,
            'V': tr,
            'stim_start': [stim_start],
            'stim_end': [stim_end]
        }
        for tr in chunk
    ]
    
    try:
        features_list = efel.getFeatureValues(trace_data_list, EFEL_FEATURE_NAMES)
    except Exception as e:
        print(f'WARNING: EFEL failed for chunk {chunk_idx}, error: {e}')
        return (chunk_idx, chunk_out)

    for j, result_dict in enumerate(features_list):
        for k, name in enumerate(EFEL_FEATURE_NAMES):
            val = result_dict.get(name)
            if val is not None and len(val) > 0:
                chunk_out[j, k] = np.mean(val)
            else:
                chunk_out[j, k] = 0.0

        ap_amp = result_dict.get('AP_amplitude')
        chunk_out[j, -1] = float(len(ap_amp)) if ap_amp is not None else 0.0
    
    return (chunk_idx, chunk_out)


def extract_efel_features_from_volts(volts, dt=0.1, chunk_size=4096, num_threads=4):
    """
    Extract EFEL features from raw voltage traces before normalization using parallel processing.

    Input shapes supported:
      - (samples, time_bins, probes, stims)
      - (samples, time_bins, probes)
    Output shape:
      - (samples, probes, stims, num_features_plus_ap_count)
    """
    if volts.ndim == 4:
        num_samples, num_time_bins, num_probes, num_stims = volts.shape
        traces = np.transpose(volts, (0, 2, 3, 1)).reshape(-1, num_time_bins)
    elif volts.ndim == 3:
        num_samples, num_time_bins, num_probes = volts.shape
        num_stims = 1
        traces = np.transpose(volts, (0, 2, 1)).reshape(-1, num_time_bins)
    else:
        raise ValueError(f"Unsupported `volts` shape for EFEL extraction: {volts.shape}")

    n_features = len(EFEL_FEATURE_NAMES) + 1  # +1 for AP count
    out = np.zeros((traces.shape[0], n_features), dtype=np.float32)

    time_array = np.arange(num_time_bins, dtype=np.float64) * dt
    stim_start = 0.0
    stim_end = float(num_time_bins) * dt

    # Prepare chunk arguments
    chunk_jobs = []
    for chunk_idx, i0 in enumerate(range(0, traces.shape[0], chunk_size)):
        i1 = min(i0 + chunk_size, traces.shape[0])
        chunk = traces[i0:i1]
        chunk_jobs.append((chunk, time_array, stim_start, stim_end, chunk_idx, i0, i1))

    print(f'extract EFEL features: traces={traces.shape[0]}, features={n_features}, '
          f'chunk_size={chunk_size}, num_chunks={len(chunk_jobs)}, num_processes={num_threads}')
    
    # Process chunks in parallel with progress bar using ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_process_efel_chunk, chunk, time_arr, stim_s, stim_e, cidx): (cidx, i0, i1) 
                   for chunk, time_arr, stim_s, stim_e, cidx, i0, i1 in chunk_jobs}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing EFEL chunks'):
            cidx, chunk_out = future.result()
            _, i0, i1 = futures[future]
            out[i0:i1] = chunk_out[:i1-i0]

    return out.reshape(num_samples, num_probes, num_stims, n_features)




#...!...!..................
def format_raw(dom,off_len,addStim=False,num_threads=4):
    [ioff,myLen]=off_len
    print('\nformat:', dom,[ioff,myLen])

    for xN in simD:
        yN='%s_%s'%(dom,xN)
        bigD[yN]=simD[xN]
        # bigD[yN]=simD[xN][ioff:ioff+myLen]

        #print('new ',dom,yN,bigD[yN].shape)

    # .... extract EFEL features from raw volts BEFORE normalization ...
    volts=bigD.pop(dom+'_volts')
    efel_features = extract_efel_features_from_volts(volts, num_threads=num_threads)
    bigD[dom+'_efel_features'] = efel_features

    # .... normalize and QA volts ...
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
  thread_total=args.thread_total
  threads_for_efel = 256
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
    #temp Remove later
    # nval=int(totSamp)
    # tval= int(nval/thread_total)
    # train_val=0
    

    # compute offest and length per domain  (Offset,Length)
    split_index={'valid':[0+thread_id*tval,tval], 'test':[nval+thread_id*tval,tval],'train':[2*nval+thread_id*thread_train_val, thread_train_val]}
    split_index_total = {'valid':[0,nval],'test':[nval,nval],'train':[2*nval,train_val]}

    #temp Remove later
    # split_index={'valid':[0,1], 'test':[0+thread_id*tval,tval],'train':[0,1]}
    # split_index_total = {'valid':[0,nval],'test':[0,nval],'train':[2*nval,train_val]}

    print('M:split_index',split_index)
    print("M:Thread_id",thread_id)
    bigD={}  # this will be output
    totFlat=0
    
    # keep this order to deal with bigest record first - it may not matter
    for dom in ['train','valid','test']: 
        simD=read3_only_data_hdf5(inpF,1,None,split_index[dom][0],split_index[dom][1])
        #... isolate stims to be not split
        tmpD={xN:simD.pop(xN) for xN in stimNL}
        
        totFlat+=format_raw(dom,split_index[dom],args.addStim,num_threads=threads_for_efel)
        
    #.... update meta data
    pmd={
        'split_index':split_index_total,
        'num_flat_volts':totFlat,
        'pack_conf':args.conf,
        'full_input_h5':inpF0,
        'efel_feature_names': EFEL_FEATURE_NAMES 
    }
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
