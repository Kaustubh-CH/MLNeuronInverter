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
    parser.add_argument("-f","--fileName",help="Name of the hdf5 stored",  default="ALL_CELLS")
    parser.add_argument("--probes", type=int,nargs='+',required=False, default=None, help="Selecting number of probes")

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
    # allL2 =allL2[:101]
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
    if nBadS!=0:
        print('clear NaN volts in %d sample(s)'%len(badIdx))
        zeroVolts=np.zeros_like( bigD['volts'][0])
        for i in  badIdx:
            bigD['volts'][i]=zeroVolts
        #exit(1)  # activate it to abort on NaN
    return nBadS

def fill_bigD_single_cell(currSimPath,jId,args):
    
    for dir_path in os.listdir(currSimPath):
        if(os.path.isdir(os.path.join(currSimPath,dir_path))):
            current_cell = dir_path
    print("working on",current_cell)
    if(current_cell not in cells):
        cells.append(current_cell)
    new_path = os.path.join(currSimPath,current_cell)
    hdf5_single_cell = os.listdir(new_path)
    # hdf5_single_cell = hdf5_single_cell[:101]
    currNFile=len(hdf5_single_cell)
    nBadSamp=0
    for j,name in enumerate(hdf5_single_cell):
        inpF=os.path.join(new_path,name)
        if j<10:  print('read',j,name)
        else: print('.',end='',flush=True)
        try:
            simD,_=read3_data_hdf5(inpF,verb=0)
        

            nBadSamp+=clear_NaN_samples(simD)
            ioff=jId*currNFile*oneSamp + j*oneSamp # m_type*num_hdf5+j*num_samples
            curr_index = indices[ioff:ioff+oneSamp]
            for x in simD:
                if(x=="unit_par" or x=="phys_par"):
                    bigD[x][curr_index]=simD[x][:,idx]
                    if(np.sum(bigD[x][curr_index]==0)>16):
                        1 # asdf #should be elimnating idx
                elif(x=="volts" and args.probes) :
                    bigD[x][curr_index]=simD[x][:,:,args.probes,:]
                else:
                    bigD[x][curr_index]=simD[x]
        except OSError:
            print("Problem with file",inpF)
    oneMD['simu_info']['num_NaN_volts_cleared']=int(nBadSamp)

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
    oneMD['simu_info']['stim_names']=nameL2

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
    # h5L=h5L[:205]
    nfileTotal= len(h5L)*len(args.jid) #Increasing for all cell assuming all cells have same number of hdf5
    idx=[]
    numExclude=0
    if('include' in oneMD.keys()):
        idx =   oneMD['include']
        _,totalPar = oneD['phys_par'].shape
        numExclude = totalPar-len(idx)
    oneSamp=oneD['volts'].shape[0]
    #Fill in meta data


    print("Total number of files Cells * hdf5s",nfileTotal)
    assemble_MD(nfileTotal)
    siz=0
    for xN in oneD:
        one=oneD[xN]
        sh1=list(one.shape)
        sh1[0]=nfileTotal*oneSamp
        if(xN=="unit_par" or xN=="phys_par"):
            siz=sh1[1]
            sh1[1]-=int(numExclude)
        if(xN=="volts" and args.probes):
            sh1[2]=len(args.probes)
        bigD[xN]=np.zeros(tuple(sh1),dtype=np.float16)  
    
    

    #import csvs

    indices=np.arange(nfileTotal*oneSamp)
    np.random.shuffle(indices)

    cells=[]
    for jId,currJid in enumerate(args.jid):
        currSimPath =  os.path.join(args.simPath,str(currJid+'_1'))
        print("reading for cell",)
        fill_bigD_single_cell(currSimPath,jId,args)
    zero_count = np.all(bigD['unit_par'][:,:]==0,axis=1)
    print(np.sum(zero_count))
    oneMD['all_cells']=cells
    import_stims_from_CVS()
    outF=os.path.join(args.outPath,args.fileName+'.simRaw.h5')
    write3_data_hdf5(bigD,outF,metaD=oneMD)