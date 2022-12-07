#!/usr/bin/env python3
'''
identify spikes in simulated  wave forms
score wavforms by counting valid spikes

'''
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_Experiment import SpikeFinder
from findSpikesExpB import score_me, pack_scores

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-d", "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-simulation/sim8kHz-as2019//',help="formated data location")

    parser.add_argument("--dataName",  default='bbp153', help="shortName for a set of routines ")
    parser.add_argument("--formatName",  default='simB.8kHz', help="data name extesion ")

    parser.add_argument("-o","--outPath", default='out2019/',help="output path for plots and tables")
    parser.add_argument("--confName",  default='conf_scoreChaos.yaml', help="algo configuration")
 
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!..................

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpF='%s.%s.h5'%(args.dataName,args.formatName)
    bigD,inpMD=read3_data_hdf5(args.dataPath+inpF, verb=0)

    timeV=bigD['time']
    spiker=SpikeFinder(timeV,verb=args.verb)

    waves2D=bigD['waveform']
    #1waves2D=waves2D[1:2,15:16]  # pick 1 good waveform, for debug
    #waves2D=waves2D[2:4,1:3]  # pick 1 for debug
    inp_shape=tuple(waves2D.shape[:-1])
    numTimeBin=inpMD['numTimeBin']
    print('M:inp_shape',inp_shape, 'waves2D:',waves2D.shape)
    spikeTraitL2,mxSpk=score_me(waves2D.reshape(-1,numTimeBin),spiker)
    traitUnits=[['tPeak','ms'],['yPeak','mV'],['twidth','ms'],['y_twidth','mV'],['twidth_at_base','ms']]
    bigD['waveform']=waves2D # save only soma-data
    totSpikes,totSweep,maxSpikes=pack_scores(spikeTraitL2,mxSpk, inp_shape,bigD,traitUnits)

    # save results
    inpMD['spikes']={'traits':traitUnits,'totSpikes':totSpikes,'maxSpikes':maxSpikes,'totWaves':len(spikeTraitL2)}
    inpMD['spikerConf']=spiker.conf
       
    outF=inpF.replace(args.formatName,'spikerSum')
    write3_data_hdf5(bigD,args.outPath+outF,metaD=inpMD)
    print('M:done %s totSpikes=%d totSpikySweep=%d'%(args.dataName,totSpikes,totSweep))

    
