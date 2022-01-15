#!/usr/bin/env python3
'''
 plot overaly of raw experimental data collected by Roy
purpose : show TEA widens spikes
'''

import sys,os
import h5py
from pprint import pprint
import copy
import numpy as np

from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False,help="disable X-term for batch mode")
    parser.add_argument("--rawPath",
                        default='/global/homes/r/roybens/fromMac/neuron_wrk/NeuronStable',
                        help="input  raw data path for experiments")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    args.formatVenue='prod'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)

#...!...!..................
    def waveform(self,plDD,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(7.5,6))
        ax = self.plt.subplot(1,1,1)

        timeV=plDD['timeV']
        N=len(plDD['waves'])

        for n in range(0,N):
            hexc='C%d'%(n%10+2)
            
            ax.plot(timeV,plDD['waves'][n], color=hexc, label=plDD['wLab'][n],linewidth=1.5)

        ax.legend(loc='best')
        tit=plDD['shortName']
        xLab='time (msec)'
        yLab='waveform (mV) '
        ax.set(title=tit,xlabel=xLab,ylabel=yLab)
        #ax.grid()
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
        if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    args.prjName='expC'

    cellName='211219_5'
    
    inpF=os.path.join(args.rawPath,cellName+'.NiExp.h5')
    print('M:rawPath=',args.rawPath,inpF)
    bulk,expMD=read3_data_hdf5(inpF)
    
    waves=bulk['Vs']  # as-is
    #waves=bulk['60HzFilteredVs']  # filtered by roy
    stim_time=bulk['stim_time']
    stims=bulk['stim_waveform']

    exp_log=expMD.pop('exp_log')
    pprint(expMD)
    for rec in exp_log:
        #
        rtn_id_start=rec.pop('rtn_id_start')
        print('rtn_id_start:',rtn_id_start)
        print("   ",rec)

    js=1  # sweep index
    mySel=[[22,'no drug'],[43,'0.25 mM TEA'],[56,'0.5 mM TEA']]
    waveL=[]
    wLab=[]
    ampl=None
    for a,b in mySel:
        waveL.append(waves[a][js]*1e3)
        wLab.append(b)
        ampl=bulk['stim_ampl'][a]

        print('select %d routine ampl=%.2f'%(a,ampl))
    # - - - - - PLOTTER - - - - -
    plDD={'waves':waveL,'wLab':wLab}
    plot=Plotter(args)
      
    plDD['timeV']=stim_time*1e3
    plDD['stim_ampl']=ampl
    plDD['shortName']=cellName+' '+str(mySel)+' ampl %.2f'%ampl

    #- - - -  display
    #plDD['timeLR']=[31.,41]  # (ms)  time range 1st spike
    plDD['timeLR']=[52.,60]  # (ms)  time range 2nd spike
    
    if 1:
        #plDD['amplLR']=[-90,70]  #  (mV) amplitude range
        sumL=plot.waveform(plDD)

    plot.display_all('rawCover')
