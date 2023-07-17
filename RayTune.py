from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tuner import Tuner
from ray import air
# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.search.dragonfly import DragonflySearch
from ray.tune.suggest.optuna import OptunaSearch
# from ray.tune.suggest.ax import AxSearch
from toolbox.Trainer import Trainer
from ray.air import session
import ray
import multiprocessing
import threading
import torch.distributed as dist


def threadTrain(params):
  #  print(params)
  #  params=dict(params)
   
  #  params['world_rank']=int(threading.get_ident())
  #  print('WORLD RANK',params['world_rank'])
   try:
      trainer = Trainer(params)
      trainer.train()
   except RuntimeError as e:
      print("RAYTUNE Exception",type(e).__name__, e.args)
      from ray.air import session
      session.report({"loss": 5})
   except Exception as E:
      import traceback
      traceback.print_exc()

def trainable(params):
    cnn_depth=params['model']['conv_block']['cnn_depth']
    # print(params['model']['conv_block']['filter'])
    params['model']['conv_block']['filter']=list(params['model']['conv_block']['filter'].values())[:cnn_depth]    
    params['model']['conv_block']['kernel']=list(params['model']['conv_block']['kernel'].values())[:cnn_depth]
    params['model']['conv_block']['pool']=list(params['model']['conv_block']['pool'].values())[:cnn_depth]
    fc_depth=params['model']['fc_block']['fc_depth']
    params['model']['fc_block']['dims']=list(params['model']['fc_block']['dims'].values())[:fc_depth]
    params['model']['fc_block']['dims'].append(128)
    print("Model Shape ",params['model']['conv_block']['filter'])
    print("Model Kernel Shape ",params['model']['conv_block']['kernel'])
    print("Model Pooling ",params['model']['conv_block']['pool'])

    try:
      trainer = Trainer(params)
      trainer.train()
    except RuntimeError as e:
      print("RAYTUNE Exception",type(e).__name__, e.args)
      from ray.air import session
      session.report({"loss": 2})
    except Exception as E:
      import traceback
      traceback.print_exc()
      from ray.air import session
      session.report({"loss": 5})
    # pool = multiprocessing.Pool(processes=params['world_size'])
    # result=[]

    # torch.cuda.set_device(0)
    # dist.init_process_group(backend='nccl', init_method='env://')
    # # params['world_rank'] = dist.get_rank()
    # for thread_id in range(params['world_size']):
    #    params['world_rank']=thread_id
    #    res= pool.apply_async(threadTrain, (params,))
    #    result.append(res)
    # pool.close()
    # pool.join()

    
      
class Raytune:


    def __init__(self,params):
        # trainer = Trainer(params)
        max_num_epochs=10
        gpus_per_trial=2
        num_samples=1000
        cpus_per_trail=8
        # trainer.train()
        scheduler = ASHAScheduler(
            # metric="loss",
            # mode="min"
            # max_t=max_num_epochs,
            # grace_period=1,
            # reduction_factor=2
            )
        # algo  = AxSearch()
        algo = OptunaSearch()
        # algo = BayesOptSearch(random_search_steps=4)
        # hyperopt_search = HyperOptSearch(
        #           metric="loss", mode="min")
        # algo = DragonflySearch()
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["loss", "accuracy", "training_iteration"])
        
        # print("Resources used by Ray",ray.cluster_resources())
        # params['model']['conv_block']['cnn_depth']=tune.randint(2, 8)
        params['model']['conv_block']['cnn_depth']=tune.choice([2,3,4,5,6,7,8])
        params['model']['conv_block']['filter']={str(x):tune.choice([30, 60, 90, 120]) for x in range(8)}
        params['model']['conv_block']['kernel']={str(x):tune.choice([1,2,3,4,5,6]) for x in range(8)}
        params['model']['conv_block']['pool']={str(x):tune.choice([1,2,3,4,5,6]) for x in range(8)}
        params['model']['fc_block']['fc_depth']=tune.choice([4,5,6,7,8])
        params['model']['fc_block']['dims']={str(x):tune.choice([256,512,768,1024]) for x in range(8)}
       
        # params['model']['conv_block']['filter']=[tune.choice([30, 60, 90, 120]) for _ in range(8)]
        # params['model']['conv_block']['kernel']=[tune.choice([3,4,5,6]) for _ in range(8)]
        # params['model']['conv_block']['pool']=[tune.choice([3,4,5,6]) for _ in range(8)]
        # params['model']['conv_block']['filter']=[tune.choice([30, 60, 90, 120]) for _ in range(8)]
        # params['model']['conv_block']['filter']=[tune.choice([30,60,90,120])]*8 #this is a blunder
        # params['model']['conv_block']['filter']=[tune.randint(1, 4) for _ in range(8)]
        # params['model']['conv_block']['kernel']=[tune.randint(3, 6) for _ in range(8)]
        # params['model']['conv_block']['pool']=[tune.randint(3, 6) for _ in range(8)]
        # tune.choice([[30, 120, 240], [30, 90, 180]])
        # params['model']['conv_block']['filter']=tune.choice([[30, 120, 240], [30, 90, 180]])


        # params['world_size']=cpus_per_trail
        
        tuner = Tuner(tune.with_resources(
                        tune.with_parameters(trainable),
                        resources={"cpu": cpus_per_trail, "gpu": gpus_per_trial}
                        ),
                      tune_config=tune.TuneConfig(
                        metric="loss",
                        mode="min",
                        scheduler=scheduler,
                        # search_alg=hyperopt_search,
                        search_alg=algo,
                        num_samples=num_samples
                        ),
                      run_config = air.RunConfig(
                        local_dir="./out"
                        ),
                      param_space = params
                      
                      )
        results = tuner.fit()
        best_result = results.get_best_result("loss", "min")
        # best_result = results.get_best_result()
        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
        best_result.metrics.keys()))
        print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
        

        # result = tune.run(
        #     train_cifar,
        #     resources_per_trial={"cpu": 1, "gpu": 2},
        #     config=config,
        #     num_samples=num_samples,
        #     scheduler=scheduler,
        #     progress_reporter=reporter)

        # best_trial = result.get_best_trial("loss", "min", "last")
        # print("Best trial config: {}".format(best_trial.config))
        # print("Best trial final validation loss: {}".format(
        #     best_trial.last_result["loss"]))
        # print("Best trial final validation accuracy: {}".format(
        #     best_trial.last_result["accuracy"]))

        # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
        # device = "cpu"
        # if torch.cuda.is_available():
        #     device = "cuda:0"
        #     if gpus_per_trial > 1:
        #         best_trained_model = nn.DataParallel(best_trained_model)
        # best_trained_model.to(device)

        # best_checkpoint_dir = best_trial.checkpoint.value
        # model_state, optimizer_state = torch.load(os.path.join(
        #     best_checkpoint_dir, "checkpoint"))
        # best_trained_model.load_state_dict(model_state)

        # test_acc = test_accuracy(best_trained_model, device)
        # print("Best trial test set accuracy: {}".format(test_acc))


    