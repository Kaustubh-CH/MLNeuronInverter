from pprint import pprint
from toolbox.Util_H5io3 import read3_data_hdf5
inpF='mlPred/bbp153-a1.00_expF2_lr0.0050.mlPred.h5'
bigD,inpMD=read3_data_hdf5(inpF)
print('inpMD'); pprint(inpMD)
