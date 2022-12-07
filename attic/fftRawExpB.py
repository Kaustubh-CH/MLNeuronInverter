#!/usr/bin/env python3
'''
 plot raw experimental data collected by Roy, a single input file
'''

import sys,os
import h5py
from pprint import pprint
import copy
import numpy as np
from scipy import signal
import matplotlib.colors as colors


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

    parser.add_argument("-r", "--routine", type=str, default='211002_1_NI_018_16', help="[.h5]  single measurement file")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='fftB'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!..................
def notch_filtering(wav, fs, w0,Q=3):
    """ Apply a notch (band-stop) filter to the audio signal.
    Args:
        wav: Waveform.
        fs: Sampling frequency of the waveform.
        w0: The frequency to filter. See scipy.signal.iirnotch.
        Q: See scipy.signal.iirnotch.
        Quality factor. Dimensionless parameter that characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw.
    Returns:
        wav: Filtered waveform.
    """

    b, a = signal.iirnotch(2 * w0/fs,Q=Q)
    wav = signal.lfilter(b, a, wav)
    return wav

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
               
#...!...!..................
    def doFFT(self,wave,plDD,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,12))

        # amplitude
        nrow,ncol=2,1
        ax = self.plt.subplot(nrow,ncol,1)
        timeV=plDD['timeV']
        T=plDD['fft_t']
        F=plDD['fft_f']
        S=plDD['fft_S']

        
        hexc='C1'
        wLab='some55'
        ax.plot(timeV,wave, color=hexc, label=wLab,linewidth=0.7)

        ax.legend(loc='best')
        tit=plDD['shortName']
        xLab='time (sec)'
        yLab='waveform (V) '
        ax.set(title=tit,xlabel=xLab,ylabel=yLab)
        ax.grid()
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))

        # fire FFT
        ax = self.plt.subplot(nrow,ncol,2)
        # find z-range
        zu=S.max(); zd=S.min()
        zd=np.sqrt(zu*zd)
        
        img = ax.pcolormesh(T, F, S,shading='auto', cmap='binary' , norm=colors.LogNorm(vmin=zd, vmax=zu) )
        ax.set(title='FFT power spectrum',xlabel=xLab,ylabel='Frequency (Hz)')
        # )
        #powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(wave, Fs=500000)
        self.plt.colorbar(img, ax=ax)
        ax.grid()
        #The darker the color of the spectrogram at a point, the stronger is the signal at that point.
        #print('alt:', powerSpectrum.shape, freqenciesFound.shape, time.shape)
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))

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
    
    xL=args.routine.split('_')
    args.rawPath+='/'+'_'.join(xL[:2])+'/'
    inpF='%s/%s.h5'%(args.rawPath,args.routine)
    print('M:rawPath=',args.rawPath,inpF)
    bulk,_=read3_data_hdf5(inpF)
    waves=bulk['Vs']
    timeV=bulk['time']
    stims=bulk['stim']
    wave=np.copy(waves[0])

    fs=50000 # sampling freq

    if 1:
        # remove some freq
        for w in [ 19, 17, 12, 10, 8 ]:
            w0=w*100  # freq to be removed
            wave=notch_filtering(wave, fs, w0,Q=3)
  
    waves[1]=wave
    f, t, Sxx = signal.spectrogram(wave, fs=fs, nfft=64,nperseg=64)

    plDD={}
    plDD['fft_t']=t
    plDD['fft_f']=f
    plDD['fft_S']=Sxx
    
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)

    plDD['shortName']=args.routine
    #for x in [ 'units',: plDD[x]=inpMD[x]
    plDD['timeV']=timeV

    #- - - -  display
    plDD['timeLR']=[0.,0.2]  # (ms)  time range 
    #plDD['amplLR']=[-90,70]  #  (mV) amplitude range
    plot.doFFT(wave,plDD)

    plot.waveform(waves,plDD)

    plot.display_all()
