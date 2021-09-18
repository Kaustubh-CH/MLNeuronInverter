#!/usr/bin/env python3
# examine raw data from Vyassa

from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
from toolbox.Util_IOfunc import read_yaml

import numpy as np
from pprint import pprint

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-o","--outPath",
                        default='out/',help="output path for plots and tables")

    parser.add_argument("-N", "--cellName", type=str, default='bbp153',
                        help="cell shortName")

    args = parser.parse_args()
    args.dataPath='/global/cfs/cdirs/m2043/balewski/neuronBBP-data_67pr/' #Cori
    args.formatVenue='prod'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args



#...!...!..................
def shorten_param_names(inpL):
    mapD={'_apical':'_api', '_axonal':'_axn','_somatic':'_som','_dend':'_den'}
    outL=[]
    print('M: shorten_param_names(), len=',len(inpL))
    for x in inpL:
        #print('0x=',x)
        for k in mapD:
            x=x.replace(k,mapD[k])
        x=x.replace('_','.')
        outL.append(x)
    return outL

#...!...!..................
def expand_param_names(inpL):
    mapInv={'_api':'_apical', '_axn':'_axonal','_som':'_somatic','_den':'_dend'}

    print('M: expand_param_names(), len=',len(inpL))
    #pprint(mapInv)
    outL=[]
    for x in inpL:
        #print('0x=',x)
        x=x.replace('.','_')
        for k in mapInv:
            x=x.replace(k,mapInv[k])
        outL.append(x)
    return outL


#............................
#............................
#............................
class Plotter_OneAAA(Plotter_Backbone):
    def __init__(self, args,metaD):
        Plotter_Backbone.__init__(self,args)
        self.metaD=metaD
#...!...!..................
    def VoltsSomaNice(self,X,stim,ifr,figId=100):

        probeNameL=self.metaD['featureName']
        nBin=X.shape[1]
        maxX=nBin ; xtit='time bins'
        binsX=np.linspace(0,maxX,nBin)
        numProbe=X.shape[-1]

        assert numProbe<=len(probeNameL)

        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,5))

        print('plot input traces, numProbe',numProbe)

        yLab='ampl (mV)'
        ax = self.plt.subplot(1,1,1)
        ipr=0
        amplV=X[ifr,:,ipr]/150.
        ax.plot(binsX,amplV,label='%d:%s'%(ipr,probeNameL[ipr]), linewidth=0.7)
        ax.plot(stim*10, label='stim', linewidth=0.7,color='black', linestyle='--')
        ax.legend(loc='best')
        ax.set(title='frame=%d'%ifr)
        #ax.set_xlim(700,1800)
        ax.grid()
        
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':

    args=get_parser()
    args.prjName='niceA'

    shortN=args.cellName
    shard=7
    inpF='%s/%s.cellSpike.data_%d.h5'%(shortN,shortN,shard)
    bulk,_=read3_data_hdf5(args.dataPath+inpF)

    nSamp=50
    frames=bulk['frames'][:nSamp]
    phys_par=bulk['phys_par'][:nSamp]
    print('inp data',frames.shape)

    metaF=args.dataPath+"/%s/meta.cellSpike_orig.yaml"%(args.cellName)
    metaD=read_yaml(metaF)
    print(metaD.keys())

    metaF=args.dataPath+"/%s/stim.chaotic_2.yaml"%(args.cellName)
    stimD=read_yaml(metaF)
    print(stimD.keys())
    stim=stimD['stimFunc']

    # export for Roy
    bigD={'stim':stim,'waveform':frames/150.,'phys_par':phys_par}
    outF='%s.cellSpike.vyassa_%d.h5'%(shortN,shard)

    inpMD=metaD['dataInfo']
    rawMD=metaD['rawInfo']

    simMD={x:inpMD[x] for x in ['featureName', 'parName','physRange'] }
    for x in ['bbpId','bbpName','stimName','timeAxis']:
        simMD[x]=rawMD[x]
    orgName=expand_param_names(simMD['parName'])
    #for a,b in zip(orgName,simMD['parName']):  print(a,'   ',b)
    simMD['parNameOrg']=orgName
    print('simMD:',sorted(simMD))
    write3_data_hdf5(bigD,args.outPath+outF,metaD=simMD,verb=1)


    # - - - -   only plotting - - - - 
    plot=Plotter_OneAAA(args,metaD['dataInfo'])
    frid=2
    plot.VoltsSomaNice(frames,stim,frid, figId=frid)
    plot.display_all('jan')
