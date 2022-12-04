#!/usr/bin/env python3
""" 

 re-pack samll hd5 NEURON output to one  6k-samples HD5 files
 ./aggreagte_Kaustubh.py  --dataPath ...


"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os,sys
from toolbox.Util_H5io3 import   read3_data_hdf5, write3_data_hdf5
from pprint import pprint
import numpy as np

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s","--simPath",help="simu raw path",  default='/pscratch/sd/k/ktub1999/BBP_TEST2/runs2/')
    parser.add_argument("-o","--outPath",help="output  path",  default='/pscratch/sd/b/balewski/tmp_bbp3')
    parser.add_argument("--jid", type=str, default='3800565_1', help="cell shortName list, blanks separated")
    
    args = parser.parse_args()
    args.verb=1
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
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


#...!...!..................
def assemble_MD():
    _,nppar=oneD['phys_par'].shape
    _,nspar,_=oneD['phys_stim_adjust'].shape
    _,ntbin,nprob,nstim=oneD['volts'].shape
    prnmL=oneMD.pop('probeName')
 
    #... append info to MD
    smd={}
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
    md['simu']=smd
    md['num_time_bins']=ntbin
    md['num_phys_par']=nppar
    md['num_stim_par']=nspar
    md['cell_name']=cellName
            
    
#...!...!..................
def import_stims_from_CVS():
    nameL2=[]
    nameL1=oneMD.pop('stimName')
    for name in  nameL1:
        inpF=os.path.join('data5',name)
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
    oneMD['simu']['stim_names']=nameL2


#...!...!..................
def read_all_h5(inpL,dom):

    nfile=len(inpL)
    oneSamp=oneD['volts'].shape[0]
    #print('aa',inpL); oko9
    a2a={}
    #... create containers
    for x in oneD:
        one=oneD[x]
        sh1=list(one.shape)
        sh1[0]=nfile*oneSamp
        print(dom,x,sh1)
        domN='%s_%s'%(x,dom)
        bigD[domN]=np.zeros(tuple(sh1),dtype=one.dtype)
        a2a[x]=bigD[domN]  # shortuct for accumulating output

    #.... main loop over data
    print('RA5:read h5...',dom,nfile)
    for j,name in enumerate(inpL):
        inpF=os.path.join(pathH5,name)
        if j<10: print('read',j,name)
        simD,_=read3_data_hdf5(inpF,verb=0)
        ioff=j*oneSamp
        for x in simD:
            a2a[x][ioff:ioff+oneSamp]=simD[x]
            
                
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    
    simPath=os.path.join(args.simPath,args.jid)
    h5L,pathH5,oneD,oneMD,cellName=get_h5_list(simPath)
    #h5L=h5L[:20]
    nh5=len(h5L)
    
    #bigD,splitL=assemble_containers()
    assemble_MD()
    bigD={}
    import_stims_from_CVS()

    # ... split data into domains
    nval=int(nh5/10)
    read_all_h5(h5L[:nval],'valid')
    read_all_h5(h5L[nval:2*nval],'test')
    read_all_h5(h5L[2*nval:],'train')

    print('M:sim meta-data');   pprint(oneMD)
    print('M:big',list(bigD))

    outF=os.path.join(args.outPath,oneMD['cell_name']+'.simRaw.h5')
    write3_data_hdf5(bigD,outF,metaD=oneMD)
    print('M:done')
    
    from toolbox.Util_IOfunc import write_yaml # for testing only
    write_yaml(oneMD,'aa.yaml')
