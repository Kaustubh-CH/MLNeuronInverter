#!/usr/bin/env python3
'''
 formats packed Vyassa simu (aka production) for bbp153
Uses advanced hd5 storage, includes meta-data in hd5

'''
import sys,os
from pprint import pprint
import numpy as np

sys.path.append(os.path.abspath("../"))
from toolbox.Util_H5io3  import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml, read_yaml

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
        
    parser.add_argument("--cellName",  default='bbp153', help="cell shortName ")
    parser.add_argument("--inpPath",help="path to input",
                        default='/global/homes/b/balewski/prjn/neuronBBP-pack8kHzRam/probe_3prB8kHz/ontra3/etype_8inhib_v1/' # inhibitory
                        #default='/global/homes/b/balewski/prjn/neuronBBP-pack8kHzRam/probe_4prB8kHz/ontra4/etype_excite_v1/' # excitaory
    )
    parser.add_argument("--outPath", default='/global/homes/b/balewski/prjn/2021-roys-simulation/vyassa8kHz/',help="output path  formatted Vyassa simu  ")
    parser.add_argument("-n", "--numSample", type=int, default=50, help="clip number of input samples")
    parser.add_argument("--probeType",  help="probe partition ",
        #default='excite'
        default='8inhib'
    )

    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    domain='val'
    metaF=args.inpPath+'%s_%s.cellSpike_3prB8kHz.meta.yaml'%(args.cellName,args.probeType)
    bulk=read_yaml(metaF,verb=1)
    inpMD=bulk['dataInfo']
    rawMD=bulk['rawInfo']
    packMD=bulk['packInfo']

    inpF=inpMD['h5name']
    skipKey=['train','test']
    bigD,_=read3_data_hdf5(args.inpPath+inpF,skipKey=skipKey)
    stimF='stim.chaotic_2_8kHz.yaml'
    stimD=read_yaml(args.inpPath+stimF,verb=1)
    stim1=stimD['stimFunc']
    waveforms=bigD[domain+'_frames']
    print('M:wafeforms',waveforms.shape,', stim1:',stim1.shape)

  
    metaD={}
    metaD['bbpName']=rawMD['bbpName']
    metaD['shortName']=rawMD['bbpId']

    #metaD['']=rawMD['']
    metaD['stimName']=rawMD['stimName']    
    metaD['units']={'stimAmpl':'FS','holdCurr':'nA','waveform':'mV','time':'ms'}
    metaD['holdCurr']=[0.]
    metaD['stimAmpl']=[1.]
    metaD['probe']=['soma']
    metaD['formatName']='simV.8kHz'
    numTime=inpMD['numTimeBin']
    
    # clip samples and take only soma data
    assert args.numSample>0
    waveforms=waveforms[:args.numSample,:,0]
    
    # restore soma ampl in mV
    waveforms=(waveforms*packMD['fnorm_std'][0] + packMD['fnorm_mean'][0])/rawMD['voltsScale']     
    numSamp=waveforms.shape[0]

    finSh=[1,args.numSample,numTime] # 1st dim is a fake holding currect
    waveforms=waveforms.reshape(finSh)
    stims=np.tile(stim1, (numSamp, 1))
    stims=stims.reshape(finSh)
    timeV=np.linspace(0,numTime,num=numTime,dtype=np.float32)*stimD['timeAxis']['step']
    print('M:fin wafeforms',waveforms.shape,', stims:',stims.shape)
    print('M:wfex',waveforms[0,7])

    bigD={'waveform':waveforms,'stim':stims,'time':timeV}
    #for x in inpConf:
        #metaD[x]=inpConf[x]
    metaD.update({'numTimeBin':numTime,'sampling': '8kHz'})

    print('M:MD');pprint(metaD)
    
    outF='%s.%s.h5'%(args.cellName,metaD['formatName'])

    write3_data_hdf5(bigD,args.outPath+outF,metaD=metaD)
    print('M:done   ')
