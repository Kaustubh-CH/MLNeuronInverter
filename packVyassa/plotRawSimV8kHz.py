#!/usr/bin/env python3
'''
 plot raw sim data from Vyassa  stored at 8 kHz
'''

import sys,os
import h5py
from pprint import pprint
import copy
import numpy as np


from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import read_yaml
from toolbox.Plotter import Plotter_NeuronInverter as Plotter

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False,help="disable X-term for batch mode")
    parser.add_argument("--rawPath",
                        #default='/global/homes/b/balewski/prjn/2021-roys-experiment/october/raw/'
                        default='/global/cscratch1/sd/balewski//neuronBBP2-packed1/data_67pr_6Kfr/'
                        ,help="input  raw data path for experiments")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("--cellName",  default='bbp1524', help="cell shortName+clone : bbpNNNC ")

    args = parser.parse_args()
    args.formatVenue='prod'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    args.prjName='rawV'
    inpF='%s/%s/%s.cellSpike.data_0.h5'%(args.rawPath,args.cellName,args.cellName)
    print('M:rawPath=',args.rawPath,inpF)
    bulk,_=read3_data_hdf5(inpF)
    #waves=bulk['Vs']
    #timeV=bulk['time']
    #stims=bulk['stim']
    X=bulk['frames']
    U=bulk['unit_par']
    P=bulk['phys_par']

    inpF='%s/%s/stim.chaotic_2.yaml'%(args.rawPath,args.cellName)
    stim= read_yaml(inpF)['stimFunc']

    inpF='%s/%s/meta.cellSpike.yaml'%(args.rawPath,args.cellName)
    inpMD= read_yaml(inpF)
    #pprint(inpMD)

    
    # - - - - - PLOTTER - - - - -
    metaD={}
    metaD['featureName']=inpMD['rawInfo']['probeName']
    for x in ['parName', 'numPar','physRange']:
        metaD[x]=inpMD['dataInfo'][x]
    sumRec={}
    sumRec['short_name']=args.cellName
    plot=Plotter(args,metaD,sumRec=sumRec)
    
    plot.frames_vsTime(X,U,8, stim=stim)
    #1plot.params1D_vyassa(p,'true P',figId=3,isPhys=True)
    #1plot.params1D(u,'true U',figId=4)    
    plot.display_all(args.cellName)
