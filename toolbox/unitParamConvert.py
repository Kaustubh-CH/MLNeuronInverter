import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import argparse

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
    
    parser.add_argument("--csvPath",  default='./unitParamsExact', help="Unit Csvs Path")
    parser.add_argument("--yamlPath",  default='./out/sum_train.yaml', help="yaml train data path")
    args = parser.parse_args()
    # args.prjName='neurInfer'
    # args.outPath+'/'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

if __name__ == '__main__':
    args = args=get_parser()
    temp_files  = ["Exact","MinMax","Default"]
    dir = args.csvPath
    for temp_file in temp_files:
        with open(args.yamlPath, 'r') as yaml_file:
            train_data = yaml.safe_load(yaml_file)
        all_params_predict=[]
        all_params_actual=[]
        for filename in os.listdir(dir):
            if filename.endswith('.csv') and 'Converted' not in filename:  # Check if the file is a CSV
                params_predict=[]
                params_actual=[]
                file_path = os.path.join(dir, filename)
                parName = train_data['input_meta']['parName']
                df = pd.read_csv(file_path)
                
                if('base_values' in train_data.keys()):
                    default_params = train_data['base_values']
                else:
                    default_params = pd.read_csv("/pscratch/sd/k/ktub1999/main/DL4neurons2/sensitivity_analysis/NewBase2/MeanParams0.csv")['Values'].tolist()
                # assert len(default_params)>max(train_data['input_meta']['include'])
                assert len(df['unit_params_predict'])==len(train_data['input_meta']['include'])
                j=0
                for i in range(len(parName)):
                    if(i in train_data['input_meta']['include']):
                        u_predict = df['unit_params_predict'][j] 
                        u_actual = df['unit_params_actual'][j]
                        pram = parName[i]
                        [uLb, uUb,_] = train_data['input_meta']["phys_par_range"][i]
                        
                        if temp_file == "MinMax":
                            if(u_predict<-1):
                                u_predict=-1
                            elif(u_predict>1):
                                u_predict=1
                        elif temp_file =="Default":
                            if(u_predict<-1 or u_predict>+1):
                                u_predict=0
                        if(pram=='cm_somatic' or pram =='cm_axonal' or pram=='cm_all' or pram=='e_pas_all'):
                                b_value = (uLb+uUb)/2
                                a_value = (uUb-uLb)/2
                                P_predict = b_value + a_value * u_predict
                                P_actual = b_value + a_value * u_actual
                        else:
                                new_base=default_params[i]*10**((uLb+uUb)/2)
                                b_value = 0
                                a_value = (uUb - uLb)/2
                                # a_value=1.0
                                P_predict = new_base*10**(b_value+a_value*u_predict)
                                P_actual = new_base*10**(b_value+a_value*u_actual)
                        j+=1        
                    else:
                        u = 0
                        P_predict = default_params[i]
                        P_actual = default_params[i]
                    params_actual.append(P_actual)
                    params_predict.append(P_predict)
                all_params_predict.append(params_predict)
                all_params_actual.append(params_actual)
        np.savetxt(dir+'/'+temp_file+"predictConverted.csv",all_params_predict)
        if(temp_file =="Exact"):
            np.savetxt(dir+'/'+temp_file+"actualConverted.csv",all_params_actual)                
