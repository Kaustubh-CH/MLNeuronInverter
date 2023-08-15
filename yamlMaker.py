import os
import yaml
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-file",  default='None', help="Yamls path from Ray Tune ")
    parser.add_argument("--out",  default='/pscratch/sd/k/ktub1999/tmpYmlModel', help="Store the updated yaml")
    parser.add_argument("--default-file",  default='/global/homes/k/ktub1999/Neuron/neuron4/neuroninverter/MultuStim', help="Store the updated yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args=get_parser()
    with open(args.yaml_file, 'r') as file:
        new_conf = yaml.safe_load(file)
    with open(args.default_file+'.hpar.yaml', 'r') as file:
        def_conf = yaml.safe_load(file)
    for k in new_conf.keys():
        if("model" in def_conf.keys()):
            if(k in def_conf["model"].keys()):
                def_conf["model"][k]=new_conf[k]
    def_conf["data_conf"]["max_glob_samples_per_epoch"]=600000
    if("data_conf" in def_conf.keys()):
        def_conf["data_conf"].pop("max_glob_samples_per_epoch",50000)
    def_conf["do_ray"]=False
    yaml_name = os.path.splitext(os.path.basename(args.yaml_file))[0]
    with open(args.out+'/'+yaml_name+'.hpar.yaml', 'w') as file:
        yaml.dump(def_conf, file)