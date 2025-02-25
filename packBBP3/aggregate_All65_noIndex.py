#!/usr/bin/env python3
""" 

 re-pack samll hd5 NEURON output to one  6k-samples HD5 files
 ./aggreagte_Kaustubh.py  --dataPath ...


"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os,sys,time
from toolbox.Util_H5io3 import   read3_data_hdf5, write3_data_hdf5
from pprint import pprint
import numpy as np
import json
import h5py

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s","--simPath",help="simu raw path",  default='/pscratch/sd/k/ktub1999/BBP_TEST2/runs2/')
    parser.add_argument("-o","--outPath",help="output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3_jan12')
    parser.add_argument("--jid", type=str, nargs='+',required=True, default='3800565 380056', help="cell shortName list, blanks separated")
    parser.add_argument("--idx", type=str,nargs='+',required=False, default=None, help="id of parameters to include")
    parser.add_argument("--numExclude", type=str,required=False, default='0', help="No of parameters to exclude")
    parser.add_argument("-f","--fileName",help="Name of the hdf5 stored",  default="ALL_CELLS")
    
    args = parser.parse_args()
    args.verb=1
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)

    return args

def write3_data_hdf5_partial(dataD,outF,metaD=None,verb=1):
    if metaD!=None:
        metaJ=json.dumps(metaD)
        #print('meta.JSON:',metaJ)
        dataD['meta.JSON']=metaJ
    
    dtvs = h5py.special_dtype(vlen=str)
    h5f = h5py.File(outF, 'w')
    if verb>0:
            print('saving data as hdf5:',outF)
            start = time.time()
    for item in dataD:
        rec=dataD[item]
        if verb>1: print('x=',item,type(rec))
        if type(rec)==str: # special case
            dset = h5f.create_dataset(item, (1,), dtype=dtvs)
            dset[0]=rec
            if verb>0:print('h5-write :',item, 'as string',dset.shape,dset.dtype)
            continue
        if type(rec)!=np.ndarray: # packs a single value in ot np-array
            rec=np.array([rec])
        if(item in ['phys_par','phys_stim_adjust','unit_par','unit_stim_adjust','volts'] ):
            shape_all = list(rec.shape)
            shape_all[0] = metaD['simu_info']['num_total_samples'] ##TODO CREATE BLANK DATASET FOR US TO LATER FILL
            h5f.create_dataset(item, shape=shape_all,dtype=rec.dtype)
            if verb>0:print('h5-write :',item, rec.shape,rec.dtype)
        else:
            h5f.create_dataset(item, data=rec)
            if verb>0:print('h5-write :',item, rec.shape,rec.dtype)
    h5f.close()
    xx=os.path.getsize(outF)/1048576
    print('closed  hdf5:',outF,' size=%.2f MB, elaT=%.1f sec'%(xx,(time.time() - start)))


def append_data_hdf5_index(bigD,outF,metaD,thread_id=0,thread_total=1):
    h5f = h5py.File(outF, 'a')
    for items in bigD:
        if(items in ['phys_par','phys_stim_adjust','unit_par','unit_stim_adjust','volts'] ):
            # rec=bigD[items]
            # if type(rec)!=np.ndarray: # packs a single value in ot np-array
            #     rec=np.array([rec])
            dataset = h5f[items]
            index=dataset.shape[0]
            len_thread=int(index/thread_total)
            #HDF5 loads that particular slice instead of the entire dataset.
            dataset[thread_id*len_thread:(thread_id+1)*len_thread]=bigD[items]
    h5f.close()




#...!...!..................
def normalize_volts(volts,name='',perProbe=True,verb=1):  # slows down the code a lot
    ''' 2020-Vyassa version:  perprobe=False <-- 1 common norm per all probes
    2023-Kaustubh version: perProbe=True <-- each probe is normalized to 1
    '''
    
    Ta = time.time()
    print('WW1',volts.shape,volts.dtype)  # last dim is stim-ID

    if not perProbe:
        shp=volts.shape
        volts=volts.reshape( shp[0], shp[1]*shp[2],-1)
        print('WW1 common',volts.shape)

    #... for breadcasting to work the 1st dim (=timeBins) must be skipped
    
    X=np.swapaxes(volts,0,1).astype(np.float32) # important for correct result
    print('WW2 perProbe=%r'%perProbe,X.shape)
                
    xm=np.mean(X,axis=0) # average over time bins
    xs=np.std(X,axis=0)

    
    #... to see indices of frames w/ volts==const
    result = np.where(xs==0)  
    xs[xs==0]=1  #hack:  for zero-value samples use mu=1 to allow scaling
    xm=-60.0951997
    xs=18.95055671
    #..... RENORMALIZE INPUT HERE 
    X=(X-xm)/xs   
    
    #... revert indices and reduce bit-size
    volts_norm=np.swapaxes(X,0,1).astype(np.float16)
    if not perProbe:
        volts_norm=volts_norm.reshape( shp[0], shp[1],shp[2],-1)
        print('restore',volts_norm.shape)
    
    elaTm=(time.time()-Ta)/60.
    print('norm, xm:',xm,'X:',X.shape,'elaT=%.2f min'%elaTm)
    
    nZer=np.sum(xs==0)
    zerA=xs==0
    print('   nZer=%d %s  : name=%s'%(nZer,xs,name))
        
    del X
    #TODO append stim in probe dimension
    
    print('WW3',volts_norm.shape,volts_norm.dtype)

    if verb>1: # report flat volts for each sample
        na,nb,nc=zerA.shape    
        for i,A in enumerate(zerA):
            if np.sum(A)==0: continue
            zSt=np.sum(A,axis=0)
            zBo=np.sum(A,axis=1)
            print('zer', i,np.sum(A),'stims:',zSt,' body:',zBo)
            #assert nZer==0  # to stop at 1st case
 
    return volts_norm,nZer
            
#...!...!..................
def get_only_directories(path):
    allL=os.listdir(path)
    print('get_glob_info len1',len(allL),allL)
    for x in allL:
        if 'L'!=x[0]: continue
        cellName=x
        break
    print('GHL: cellName',cellName)
    path2=os.path.join(path,cellName)
    allL2=os.listdir(path2)
    print('get_glob_info len2',len(allL2))
    return allL2

def get_h5_list(path):
    allL=os.listdir(path)
    print('get_glob_info len1',len(allL),allL)
    for x in allL:
        if 'L'!=x[0]: continue
        cellName=x
        break
    print('GHL: cellName',cellName)
    path2=os.path.join(path,cellName)
    allL2=os.listdir(path2)
    print('get_glob_info len2',len(allL2))
    #print(allL2[:10]); ok9
    
    # get meta data
    inpF=os.path.join(path2,allL2[0])
    simD,simMD=read3_data_hdf5(inpF)
    print('GHL:sim meta-data');   pprint(simMD)
    return allL2,path2,simD,simMD,cellName


#...!...!..................
def assemble_MD(nh5,tmpnSamp=0):
    _,nppar=oneD['phys_par'].shape
    _,nspar,_=oneD['phys_stim_adjust'].shape
    nSamp,ntbin,nprob,nstim=oneD['volts'].shape
    prnmL=oneMD.pop('probeName')
    
    #... append info to MD
    smd={}
    smd['num_total_samples']=nh5*nSamp
    smd['num_sim_files']=nh5
    smd['sim_path']=pathH5
    smd['full_prob_names']=prnmL
    smd['bbp_name_clone']=oneMD.pop('bbpName')   
    smd['job_id']=oneMD.pop('jobId')
    smd['NEURON_ver']=oneMD.pop('neuronSimVer')
    smd['num_probs']=nprob
    smd['num_stims']=nstim
    smd['probe_names']=[x.split('_')[0] for x in prnmL]
          
    md=oneMD
    md['simu_info']=smd
    md['num_time_bins']=ntbin
    md['num_phys_par']=nppar
    md['num_varied_phys_par']=nppar-numExclude
    md['num_stim_par']=nspar
    md['cell_name']=cellName

    #... add units to ranges
    pparRange=md.pop('physParRange')
    for i in range(nppar):
        u=['S/cm2']
        if 'cm_' in md['parName'][i]: u=['uF/cm2']
        if 'e_' in md['parName'][i]: u=['mV']
        pparRange[i]+=u
    md['phys_par_range']=pparRange
    
    sparRange=md.pop('stimParRange')
    for i in range(nspar):
        sparRange[i]+=['nA']
    md['stim_par_range']=sparRange
        
#...!...!..................
def import_stims_from_CVS():
    nameL2=[]
    nameL1=oneMD.pop('stimName')
    #stimPath='/global/cscratch1/sd/ktub1999/stims/'
    stimPath='/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/'
    for name in  nameL1:
        inpF=os.path.join(stimPath,name)
        print('import ', inpF)
        fd = open(inpF, 'r')
        lines = fd.readlines()
        #print('aaa', len(lines), type(lines[0]))
        vals=[float(x) for x in lines ]
        data=np.array( vals, dtype=np.float32)
        #print('ddd',data.shape, data[::100],data.dtype)
        name2=name[:-4]
        nameL2.append(name2)
        bigD[name2]=data[1000:]
        assert bigD[name2].shape[0]==oneMD['num_time_bins']
        
    # ... store results in containers
    oneMD['simu_info']['stim_names']=nameL2


#...!...!..................
def read_all_h5(inpL,args,idx,numExclude,new_path):
    nfile=len(inpL)
    oneSamp=oneD['volts'].shape[0]
    #print('aa',inpL); oko9

    #... create containers
    siz=0
    for xN in oneD:
        one=oneD[xN]
        sh1=list(one.shape)
        sh1[0]=nfile*oneSamp
        
        if(xN=="unit_par" or xN=="phys_par"):
            siz=sh1[1]
            sh1[1]-=int(numExclude)
        bigD[xN]=np.zeros(tuple(sh1),dtype=one.dtype)       

    nBadSamp=0
     # idx=[0,1,2,3,4,5,6,7,8,9,10,13,14,15]
    if len(idx)==0:
        idx=range(siz)
    # if(args.idx is not None):
    #     idx=args.idx
    #     idx=[int(i) for i in idx]
    # idx=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15]
    #.... main loop over data
    pathH5 = new_path
    print('RA5:read h5...',nfile)
    for j,name in enumerate(inpL):
        inpF=os.path.join(pathH5,name)
        if j<10:  print('read',j,name)
        else: print('.',end='',flush=True)
        simD,_=read3_data_hdf5(inpF,verb=0)
        #simD['volts'][1,1,1,1]=float("nan")  # code testing
        nBadSamp+=clear_NaN_samples(simD)
        
        ioff=j*oneSamp
        for x in simD:
            if(x=="unit_par" or x=="phys_par"):
                bigD[x][ioff:ioff+oneSamp]=simD[x][:,idx]
            else:
                bigD[x][ioff:ioff+oneSamp]=simD[x]
    oneMD['simu_info']['num_NaN_volts_cleared']=oneMD['simu_info']['num_NaN_volts_cleared']+int(nBadSamp)
        
#...!...!..................
def clear_NaN_samples(bigD):  # replace volts w/ 0s, only for volts
    #bigD['volts'][1,1,1,1]=float("nan")  # code testing
    #bigD['volts'][5,1,1,1]=float("nan")  # code testing
    for x in bigD:
        data=bigD[x]
        nanA= np.isnan(data)
        nBadS=0
        if nanA.any():
            nanS=np.sum(nanA,axis=(1,2,3))            
            maskS=nanS>0
            nBadS=np.sum(maskS)  # it is a tuple
            badIdx=np.nonzero(maskS)
            print('\nWARN see %d NaN bad samples in %s :'%(nBadS,x),badIdx)
            assert x=='volts'
            
    if nBadS!=0:
        print('clear NaN volts in %d sample(s)'%len(badIdx))
        zeroVolts=np.zeros_like( bigD['volts'][0])
        for i in  badIdx:
            bigD['volts'][i]=zeroVolts
        #exit(1)  # activate it to abort on NaN
    return nBadS
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    
    simPath=os.path.join(args.simPath,str(args.jid[0]+'_1'))

    h5L,pathH5,oneD,oneMD,cellName=get_h5_list(simPath)
    h5L=h5L[:205]  # shorten input for testing
    nh5=len(h5L)
    nfileTotal= len(h5L)*len(args.jid)
   
    idx=[]
    numExclude=0
    _,totalPar = oneD['phys_par'].shape
    oneMD['include']=[x for x in range(0,totalPar)]
    if('include' in oneMD.keys()):
        idx =   oneMD['include']
        _,totalPar = oneD['phys_par'].shape
        numExclude = totalPar-len(idx)
    print("Included",idx)
    print("Num Excluded",numExclude)
    print("Total number of files Cells * hdf5s",nfileTotal)
    assemble_MD(nfileTotal)
    bigD={}
    cells=[]
    oneMD['simu_info']['num_NaN_volts_cleared']=0
    for jId,currJid in enumerate(args.jid):
        bigD={}
        currSimPath =  os.path.join(args.simPath,str(currJid+'_1'))
        for dir_path in os.listdir(currSimPath):
            if(os.path.isdir(os.path.join(currSimPath,dir_path))):
                current_cell = dir_path
        print("working on",current_cell)
        if(current_cell not in cells):
            cells.append(current_cell)
        new_path = os.path.join(currSimPath,current_cell)
        hdf5_single_cell = os.listdir(new_path)
        # if(hdf5_single_cell)
        hdf5_single_cell=hdf5_single_cell[:205]
        read_all_h5(hdf5_single_cell,args,idx,numExclude,new_path)
        if 1:
            from toolbox.Util_IOfunc import write_yaml # for testing only
            write_yaml(oneMD,'aa.yaml')
        # outF=os.path.join(args.outPath,oneMD['cell_name']+'.simRaw.h5')
        outF=os.path.join(args.outPath,args.fileName+'.simRaw.h5')
        if(jId==0):
            import_stims_from_CVS()
            write3_data_hdf5_partial(bigD,outF,metaD=oneMD)
        # oneMD['all_cells']=cells
        append_data_hdf5_index(bigD,outF,oneMD,jId,len(args.jid)) 

    oneMD['all_cells']=cells


    dtvs = h5py.special_dtype(vlen=str)
    metaJ=json.dumps(oneMD)
    h5f = h5py.File(outF, 'a')
    del h5f['meta.JSON']
    dset = h5f.create_dataset('meta.JSON', (1,), dtype=dtvs)
    dset[0]=metaJ
    h5f.close()

    print("M:done")

 
