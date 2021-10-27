#!/usr/bin/env python3
""" 
format training data, save as hd5
 ./format_CellSpike.py  --dataPath data_bbp8v4_1_5h

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

#import sys,os
#sys.path.append(os.path.abspath("../toolbox"))
from toolbox.Util_IOfunc import write_yaml, read_yaml, read_one_csv, write_one_csv, expand_dash_list
from Pack_Func import agregate_raw_data
from pprint import pprint

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")

    parser.add_argument("-d","--dataPath",help="output path",  default='data')
    
    args = parser.parse_args()
    args.prjName='cellSpike'
    args.verb=1
    args.dataPath+='/'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#............................
def shorten_param_names(rawMeta):
    mapD={'_apical':'_api', '_axonal':'_axn','_somatic':'_som','_dend':'_den'}
    inpL=rawMeta['varParL']
    outL=[]
    print('M: shorten_param_names(), len=',len(inpL))
    for x in inpL:
        #print('0x=',x)
        for k in mapD:
            x=x.replace(k,mapD[k])
        x=x.replace('_','.')
        
        outL.append(x)
    rawMeta['varParL']=outL
          

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    rawMF=args.dataPath+'/rawMeta.'+args.prjName+'.yaml'
    metaF=args.dataPath+'/meta.'+args.prjName+'.yaml'

    rawMeta=read_yaml(rawMF)
    args.targetName='%s.%s.data'%(rawMeta['bbpId'],args.prjName)
    shorten_param_names(rawMeta)

    # refine meta-data
    nProbes=len(rawMeta['probeName'])
    rawMeta['num_probes'] =nProbes
    xL=rawMeta['rawJobIdS'].split(',')
    jobL=expand_dash_list(xL)
    print('M: expanded %d job(s) list'%len(jobL),jobL)
    rawMeta['rawJobIdL']=jobL
    print('pp',rawMeta['probeName'],nProbes)
    
    metaD,stimul=agregate_raw_data(args,rawMeta)
    write_yaml(metaD,metaF)
    print('M: main task completed, metaF=',metaF)
    print('M: meta content:', list(metaD['dataInfo'].keys()))

    rawD=metaD['rawInfo']
    metaD=metaD['dataInfo']
    stimF=args.dataPath+'stim.%s.yaml'%rawD['stimName']
    outD={'timeAxis': rawD['timeAxis'],'stimName':rawD['stimName'],'stimFunc':stimul}
    write_yaml(outD,stimF)

    # update summary CVS
    inpF='BBP2-head.csv'
    mapT,labT=read_one_csv(inpF)
    rec=mapT[0]
        
    rec['short_name']=rawD['bbpId']
    rec['bbp_name']=rawD['bbpName']
    rec['clone_id']=rawD['cloneId']

    rec['num_sim_frame']=metaD['totalGoodFrames']
    rec['num_hd5_file']=metaD['numDataFiles']
    rec['num_prob']=rawD['num_probes']
    # approximate size
    rec['size_hd5_GB']=int( rec['num_hd5_file'] * rawD['numFramesPerOutput'] * rawD['num_probes']/67. / 983. )
    
    outF=args.dataPath+'%s.pack8k.csv'%rec['short_name']
    write_one_csv(outF,mapT,labT)
    pprint(mapT)
    print('M:done')
