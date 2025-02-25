import h5py
import numpy as np
import argparse
import yaml
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--facility", default='corigpu', type=str)
    # parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")

    # parser.add_argument("-m","--modelPath",  default='/global/cscratch1/sd/balewski/tmp_digitalMind/neuInv/manual/', help="trained model ")
    # parser.add_argument("--dom",default='test', help="domain is the dataset for which predictions are made, typically: test")

    # parser.add_argument("-o", "--outPath", default='same',help="output path for plots and tables")
 
    # parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False, help="disable X-term for batch mode")

    # parser.add_argument("-n", "--numSamples", type=int, default=None, help="limit samples to predict")
    # parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')

    # parser.add_argument("--expFile",  default='/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/', help="Experimental data path")
    
    # parser.add_argument("--saveFile",  default='./unitParams', help="Experimental data path")
    # parser.add_argument("--cellName", type=str, default=None, help="alternative data file name ")
    # parser.add_argument("--predictStim",default=0 , type=int, nargs='+', help="list of stims, space separated")
    # parser.add_argument("--idx", nargs='*' ,required=False,type=str, default=None, help="Included parameters")
    # parser.add_argument("--test-plot-size",default=None , type=int, help="size of test plot")
    
    parser.add_argument("--h5pyPath",  default='./unitParamsExact', help="Unit Csvs Path")
    parser.add_argument("--yamlPath",  default='./out/sum_train.yaml', help="yaml train data path")
    parser.add_argument("--numCSV", type=int, default=8, help="number of CSVs to divide into")
    parser.add_argument("--saveDir", type=str, default='./', help="place to store the csvs")
    args = parser.parse_args()
    # args.prjName='neurInfer'
    # args.outPath+'/'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

if __name__ == '__main__':
    args = args=get_parser()
    with h5py.File(args.h5pyPath, 'r') as hdf:
        ground_truth_units = hdf['ground_truth_upar'][:]
        predicted_units = hdf['predict_upar'][:]
    with open(args.yamlPath, 'r') as yaml_file:
            train_data = yaml.safe_load(yaml_file)
    parName = train_data['input_meta']['parName']
    P_predict_big = np.zeros((predicted_units.shape[0],len(parName)))
    P_actual_big = np.zeros((ground_truth_units.shape[0],len(parName)))
    if('base_values' in train_data.keys()):
        default_params = train_data['base_values']
    else:
        default_params = pd.read_csv("/pscratch/sd/k/ktub1999/main/DL4neurons2/sensitivity_analysis/NewBase2/MeanParams0.csv")['Values'].tolist()
    # assert len(default_params)>max(train_data['input_meta']['include'])
    # assert ground_truth_units.shape[1]>max(train_data['input_meta']['include'])
    j =0# value in param_set included
    for i in range(len(parName)):
        if(i in train_data['input_meta']['include']):
                u_par_predict = predicted_units[:,j]
                u_par_actual = ground_truth_units[:,j]
                pram = parName[i]
                [uLb, uUb,_] = train_data['input_meta']["phys_par_range"][i]
                if(pram=='cm_somatic' or pram =='cm_axonal' or pram=='cm_all' or pram=='e_pas_all'):
                    b_value = (uLb+uUb)/2
                    a_value = (uUb-uLb)/2
                    p_predict = b_value+a_value * u_par_predict
                    p_actual = b_value+a_value * u_par_actual
                else:
                    new_base=default_params[i]*10**((uLb+uUb)/2)
                    b_value = 0
                    a_value = (uUb - uLb)/2
                    p_predict = new_base*10**(b_value+a_value*u_par_predict)
                    p_actual = new_base*10**(b_value+a_value*u_par_actual)
                P_predict_big[:,i] = p_predict
                P_actual_big[:,i] = p_actual   
                j+=1
        else:
                P_predict_big[:,i] = default_params[i]
                P_actual_big[:,i] = default_params[i]
    per_csv = ground_truth_units.shape[0]//args.numCSV
    remain_csv = ground_truth_units.shape[0]%args.numCSV
    for i in range(args.numCSV):
        if(i==args.numCSV-1):
              per_csv+=remain_csv
        np.savetxt(args.saveDir+"/predictConverted"+str(i+1)+".csv",P_predict_big[i*per_csv:(i+1)*per_csv])
        np.savetxt(args.saveDir+"/actualConverted"+str(i+1)+".csv",P_actual_big[i*per_csv:(i+1)*per_csv])
