import os,sys,time
from toolbox.Util_H5io3 import   read3_data_hdf5, write3_data_hdf5
from pprint import pprint
import numpy as np

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--simPath",help="simu raw path",  default='/pscratch/sd/k/ktub1999/BBP_TEST2/runs2/')
    parser.add_argument("-o","--outPath",help="output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3_jan12')
    parser.add_argument("--jid", type=str, nargs='+',required=True, default='3800565 380056', help="cell shortName list, blanks separated")
    parser.add_argument("--idx", type=str,nargs='+',required=False, default=None, help="id of parameters to include")
    parser.add_argument("--numExclude", type=str,required=False, default='0', help="No of parameters to exclude")
    
    args = parser.parse_args()
    args.verb=1
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)

    return args

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


def fill_bigD_single_cell(currSimPath,currJid,args):
    
    for dir_path in os.listdir(currSimPath):
        if(os.path.isdir(dir_path)):
            current_cell = dir_path
    new_path = os.path.join(currSimPath)
    hdf5_single_cell = os.listdir(new_path)
    currNFile=len(hdf5_single_cell)
    for j,name in enumerate(hdf5_single_cell):
        inpF=os.path.join(new_path,name)
        if j<10:  print('read',j,name)
        else: print('.',end='',flush=True)
        simD,_=read3_data_hdf5(inpF,verb=0)
        nBadSamp+=clear_NaN_samples(simD)
        ioff=currJid*currNFile + j*oneSamp # m_type*num_hdf5+j*num_samples
        for x in simD:
            if(x=="unit_par" or x=="phys_par"):
                bigD[x][ioff:ioff+oneSamp]=simD[x][:,idx]
            else:
                bigD[x][ioff:ioff+oneSamp]=simD[x]
    oneMD['simu_info']['num_NaN_volts_cleared']=int(nBadSamp)

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

if __name__=="__main__":
    args=get_parser()

    bigD={}
    metaD={} 
    assert len(args.jid)>0
    simPath=os.path.join(args.simPath,str(args.jid[0]+'_1'))
    
    h5L,pathH5,oneD,oneMD,cellName=get_h5_list(simPath)
    nfileTotal= len(h5L)*len(args.jid) #Increasing for all cell assuming all cells have same number of hdf5
    oneSamp=oneD['volts'].shape[0]
    #Fill in meta data
    assemble_MD(nfileTotal)
    siz=0
    for xN in oneD:
        one=oneD[xN]
        sh1=list(one.shape)
        sh1[0]=nfileTotal*oneSamp
        if(xN=="unit_par" or xN=="phys_par"):
            siz=sh1[1]
            sh1[1]-=int(args.numExclude)
        
        bigD[xN]=np.zeros(tuple(sh1),dtype=np.float16)  
    nBadSamp=0
    idx=range(siz)
    if(args.idx is not None):
        idx=args.idx
        idx=[int(i) for i in idx]

    #import csvs



    for currJid in args.jid:
        currSimPath =  os.path.join(args.simPath,str(currJid+'_1'))
        print("reading for cell",)
        fill_bigD_single_cell(currSimPath,currJid,args)

    outF=os.path.join(args.outPath,'ALL_CELLS'+'.simRaw.h5')
    write3_data_hdf5(bigD,outF,metaD=oneMD)