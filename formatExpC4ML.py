#!/usr/bin/env python3
'''
select subset of experimental waveforms and re-pack them for ML-predictions 

'''
from pprint import pprint

from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml, read_yaml
from toolbox.Plotter_Backbone import Plotter_Backbone

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-d", "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-experiment/december/data8kHz/',help="formated input data location")

    parser.add_argument("--dataName",  default='211219_5x', help="shortName for a set of routines ")

    parser.add_argument("--ampl",  default='0.14',type=str, help="(FS) amplitude ")
    parser.add_argument("--numPredConduct",  default=15,type=int, help="number of predicted conductances (needed by ML)")

    parser.add_argument("-o","--outPath", default='exp4ml/',help="output path for plots and tables")
    parser.add_argument( "-X","--noXterm", dest='noXterm',
                         action='store_true', default=False,help="disable X-term for batch mode")

    args = parser.parse_args()
    args.formatName='expC.8kHz'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
#...!...!..................
    def waveform(self,waves,plDD,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,8))
        ax = self.plt.subplot(1,1,1)

        timeV=plDD['timeV']
        N=waves.shape[0]

        for n in range(0,N):
            hexc='C%d'%(n%10)
            a,b,_=plDD['traits'][n]
            dLab='%d  %.1f  %d'%(n,a,b)  
            ax.plot(timeV,waves[n], color=hexc, label=dLab,linewidth=0.7)

        ax.legend(loc='best',title='idx,routine,time(s)')
        tit=plDD['shortName']
        xLab='stim time (%s)'%plDD['units']['stimTime']
        yLab='soma waveform (%s)'%plDD['units']['somaWaveform']

        ax.set(title=tit,xlabel=xLab,ylabel=yLab)
        ax.grid()
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
        if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    inpF='%s.%s.h5'%(args.dataName,args.formatName)
    bigD,expMD=read3_data_hdf5(args.dataPath+inpF, verb=1)
    pprint(expMD)

    exp_info=expMD['expInfo']
    ampls=exp_info['amps']
            
    #k=list(exp_info['map_idx'])
    #print('k',k,args.ampl,type(args.ampl))
    i0,i1=exp_info['map_idx'][args.ampl]
    nSweep=i1-i0

    timeV=bigD['time']
    #ampls=inpMD['stimAmpl']
    waves=bigD['soma_wave'][i0:i1]
    stims=bigD['stim_wave'][i0:i1]
    traits=exp_info['sweep_trait'][i0:i1]
     
    waves=np.expand_dims(waves,2) # add 1-channel index for ML compatibility
    shortName='%s-A%s'%(args.dataName,args.ampl)
    # do NOT normalize waveforms - Dataloader uses per-wavform normalization

    if 0: # skip some waveforms : cherry picking for proposal 2021-10
        useL= [0, 1, 3, 4, 7, 8, 9]  #  'a' = no drug
        print('ww-in',waves.shape)        
        print('useL=',useL)
        waves=waves[useL]
        shortName+='s'
        if 1:
            nSweep=1
            sweepTrait=sweepTrait[:1]
            stims=stims[:1]
            waves=np.mean(waves,axis=0)
            waves=np.expand_dims(waves,0) 
        print('ww',waves.shape)
        
    
    #add fake Y
    unitStar=np.zeros((nSweep,args.numPredConduct))
    print('use ampl=',args.ampl,nSweep,waves.shape,unitStar.shape)

    #... assemble output meta-data
    keyL=['formatName','numTimeBin','rawDataPath','sampling','stimName']
    outMD={ k:expMD[k] for k in keyL}
    outMD['stimAmpl']=float(args.ampl)
    outMD['normWaveform']=True
    outMD['numSweep']=int(nSweep)
    outMD['comment']='added fake unitStar_par for consistency with simulations'
    outMD['shortName']=shortName
    outMD['sweep_trait']=traits
    
    # ... wrap it up
    outF=shortName+'.h5'
    bigD={'exper_frames':waves,'exper_unitStar_par':unitStar,'time':timeV,'stims':stims}
    
    bigD['stims']=stims
    write3_data_hdf5(bigD,args.outPath+outF,metaD=outMD)
    print('M:done,\n outMD:')
    pprint(outMD)

    
    # - - - - - PLOTTER - - - - -
    args.formatVenue='prod'
    args.prjName=shortName
    plot=Plotter(args)
    plDD={}
    plDD['shortName']=shortName
    plDD['timeV']=timeV
    plDD['traits']=traits
    plDD['units']=expMD['units']
    plDD['timeLR']=[0.,200]  # (ms)  time range 
    plot.waveform(waves,plDD)
    plot.display_all('4ML')
    
