#!/usr/bin/env python3
import sys,os
'''
converts probe names computed by Roy:
L23_PC_cADpyr_1:
- [0, 'cADpyr229_L23_PC_5ecbf9b163[0].soma[0]', 0]
- [35, 'cADpyr229_L23_PC_5ecbf9b163[0].axon[8]', 138.16587024701502]
- [6, 'cADpyr229_L23_PC_5ecbf9b163[0].apic[12]', 134.5742999935683]
- [40, 'cADpyr229_L23_PC_5ecbf9b163[0].dend[12]', 48.336495998631385]

to the format expected by Jan:
  bbp0541: [[0, 'soma_0', 8.3], [35, 'axon_8', 138.2], [6, 'apic_12', 134.6], [40, 'dend_12', 48.3]]


'''

import numpy as np
import time
from pprint import pprint

from toolbox.Util_IOfunc import read_yaml
import csv

#...!...!..................
def do_clone_split(mapL):  # generates cell-clone split to preserve 1 clone from each cell 
    import random
    praL=[]; witL=[]
    for shortB,bbpB in mapL:
        L=['%s%d'%(shortB,c)   for c in  [1,2,3,4,5] ]
        random.shuffle(L)
        print(L)
        praL+=L[:4]
        witL.append(L[4])
    print('praL',', '.join(praL),len(praL))
    print('witL',', '.join(witL),len(witL))
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
mapF='../packVyassa/BBP2-jobs3.csv'

mapL=[]
with open(mapF, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        #print(row)
        mapL.append(row[:2])
print('mapL',len(mapL),mapL)
royD=read_yaml('all_pobes_for_ontra4_5clones.yaml')
print('output\n')

#do_clone_split(mapL); ok00

j=0
for shortB,bbpB in mapL:
    for c in range(1,6):
        shortN='%s%d'%(shortB,c)
        bbpN='%s_%d'%(bbpB[:-3],c)
        if 'bbp2081'  in shortN: continue  # there is just no apical dendrite in BBP model
        if 'bbp2082' in shortN : continue
        #print('do: ',j,shortN,bbpN);
        cell=royD[bbpN]
        #print(cell);
        out=[]
        for rec in cell:
            x=rec[1]
            y=x.split('.')[-1]
            y=y.replace('[','_')
            probN=y[:-1]
            out1=[ rec[0], probN, float('%.1f'%rec[2])]
            #print(out)
            out.append(out1)
        print('  %s: '%shortN,out)
        #print('sleep 3; sbatch batchOntraTransform.slr ',shortN)
        j+=1
print('done',j)
