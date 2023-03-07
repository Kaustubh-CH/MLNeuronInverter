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


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s","--simPath",help="simu raw path",  default='/pscratch/sd/k/ktub1999/BBP_TEST2/runs2/')
    parser.add_argument("-o","--outPath",help="output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3_jan12')
    parser.add_argument("--jid", type=str, default='3800565_1', help="cell shortName list, blanks separated")
    
    args = parser.parse_args()
    args.verb=1
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)

    return args

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

    #..... RENORMALIZE INPUT HERE 
    X=(X-xm)/xs   
    
    #... revert indices and reduce bit-size
    volts_norm=np.swapaxes(X,0,1).astype(np.float16)
    if not perProbe:
        volts_norm=volts_norm.reshape( shp[0], shp[1],shp[2],-1)
        print('restore',volts_norm.shape)
    
    elaTm=(time.time()-Ta)/60.
    print('norm, xm:',xm.shape,'X:',X.shape,'elaT=%.2f min'%elaTm)
    
    nZer=np.sum(xs==0)
    zerA=xs==0
    print('   nZer=%d %s  : name=%s'%(nZer,xs.shape,name))
        
    del X
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
def assemble_MD(nh5):
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
    stimPath='/global/homes/b/balewski/neuronInverter/packBBP3/stims_dec26'
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
def read_all_h5(inpL):
    nfile=len(inpL)
    oneSamp=oneD['volts'].shape[0]
    #print('aa',inpL); oko9

    #... create containers
    for xN in oneD:
        one=oneD[xN]
        sh1=list(one.shape)
        sh1[0]=nfile*oneSamp
        bigD[xN]=np.zeros(tuple(sh1),dtype=one.dtype)       

    nBadSamp=0
    #.... main loop over data
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
            bigD[x][ioff:ioff+oneSamp]=simD[x]
    oneMD['simu_info']['num_NaN_volts_cleared']=int(nBadSamp)
        
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
    
    simPath=os.path.join(args.simPath,args.jid)
    h5L,pathH5,oneD,oneMD,cellName=get_h5_list(simPath)
    #h5L=h5L[:20]  # shorten input for testing
    nh5=len(h5L)
      
    assemble_MD(nh5)
    bigD={}
    read_all_h5(h5L)
    import_stims_from_CVS()

    print('M:sim meta-data');   pprint(oneMD)
    print('M:big',list(bigD))

    if 1:
        from toolbox.Util_IOfunc import write_yaml # for testing only
        write_yaml(oneMD,'aa.yaml')

    outF=os.path.join(args.outPath,oneMD['cell_name']+'.simRaw.h5')
    write3_data_hdf5(bigD,outF,metaD=oneMD)
    print('M:done')

 
