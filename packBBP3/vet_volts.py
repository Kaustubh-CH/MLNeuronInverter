#!/usr/bin/env python3
'''
plot BBP3 simulation: soma volts

'''
from pprint import pprint
from toolbox.Util_H5io3 import   read3_data_hdf5
import sys,os,time
import numpy as np


   

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    dataPath='/global/cfs/cdirs/m2043/balewski/neuronBBP3-10kHz_3pr/try2'
    dataName='L6_TPC_L1cADpyr4.simRaw.h5'
    inpF=os.path.join(dataPath,dataName)
    simD,simMD=read3_data_hdf5(inpF,skipKey=['volts_train','volts_valid'])
    #print('M:sim meta-data');   pprint(simMD)

    dom='volts_test'
    data=simD[dom]
    #data=data[:1500]
    print('\nMinp',dom,data.shape, data.dtype)
    tag_zeros(data,dom)

    print('M:done')
