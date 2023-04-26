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

from toolbox.Trainer import Trainer

def trainable(params):
    trainer = Trainer(params)
    trainer.train()

class Raytune:


    def __init__(self,params):
        trainer = Trainer(params)
        max_num_epochs=10
        gpus_per_trial=1
        num_samples=5
        # trainer.train()
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["loss", "accuracy", "training_iteration"])
        tuner = Tuner(tune.with_resources(
                        tune.with_parameters(trainable),
                        resources={"cpu": 2, "gpu": gpus_per_trial}
                        ),
                      tune_config=tune.TuneConfig(
                        metric="loss",
                        mode="min",
                        scheduler=scheduler,
                        num_samples=num_samples,
                        ),
                      param_space = params
                      
                      )
        results = tuner.fit()
        best_result = results.get_best_result("loss", "min")
        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
        print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

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


    if __name__ == "__main__":
        # You can change the number of GPUs per trial here:
        main()