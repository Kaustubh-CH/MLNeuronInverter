#!/usr/bin/env python3
'''
 plot raw experimental data collected by Roy, a single input file
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
    parser.add_argument("--rawPath", default='/global/homes/b/balewski/prjn/2021-roys-experiment/october/raw/',help="input  raw data path for experiments")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

    parser.add_argument("-r", "--routine", type=str, default='210611_6_NI_025_85', help="[.h5]  single measurement file")

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
        #self.cL10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 10 distinct colors
        
        
#...!...!..................
    def stims(self,stims,plDD,presc=1,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,8))
        ax = self.plt.subplot(1,1,1)
        pprint(plDD)
        timeV=plDD['timeV']
        
        N=stims.shape[0]
        for n in range(0,N,presc):
            hexc='C%d'%(n%10)
            ax.plot(timeV,stims[n], color=hexc, label='%d'%n,linewidth=0.7)

        ax.legend(loc='best',title='stim ampl')
        tit=plDD['shortName']
        xLab='time (s)'
        
        ax.set(title=tit,xlabel=xLab,ylabel='stim (A)')
        ax.grid()
        if 'zoom_ms' in plDD:
            [a,b]=plDD['zoom_ms']
            ax.set_xlim(a,b)

#...!...!..................
    def waveform(self,waves,plDD,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,8))
        ax = self.plt.subplot(1,1,1)

        timeV=plDD['timeV']
        N=waves.shape[0]

        for n in range(0,N):
            hexc='C%d'%(n%10)
            ax.plot(timeV,waves[n], color=hexc, label='%d'%n,linewidth=0.7)

        ax.legend(loc='best')
        tit=plDD['shortName']
        xLab='time (sec)'
        yLab='waveform (V) '
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
    args.prjName='expA'
    xL=args.routine.split('_')
    args.rawPath+='/'+'_'.join(xL[:2])+'/'
    inpF='%s/%s.h5'%(args.rawPath,args.routine)
    print('M:rawPath=',args.rawPath,inpF)
    bulk,_=read3_data_hdf5(inpF)
    waves=bulk['Vs']
    timeV=bulk['time']
    stims=bulk['stim']
    
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    plDD['shortName']=args.routine
    #for x in [ 'units',: plDD[x]=inpMD[x]
    plDD['timeV']=timeV

    #- - - -  display
    #plDD['timeLR']=[10.,160.]  # (ms)  time range 
    if 1:
        plot.stims(stims,plDD,presc=3)
    if 1:
        #plDD['amplLR']=[-90,70]  #  (mV) amplitude range
        sumL=plot.waveform(waves,plDD)

    plot.display_all('rawB')
