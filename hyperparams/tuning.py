from optuna.samplers import RandomSampler
import optuna
import random
import torch
import numpy as np
import utils.utils as utils
from data.data_generation import get_datasets
from models.gmsm_instantiations import GMSM_binary

def run_hyper_tuning(config_hyper, config_data):
    # Load data
    seed = config_hyper["seed"]
    _ = set_seeds(seed)
    datasets = get_datasets(config_data)
    models = config_hyper["models"]
    tuning_ranges = config_hyper["tuning_ranges"]
    hyper_path = "/hyperparams/" + config_hyper["name"] + "/params/"

    # Tune
    for model in models:
        tune_sampler = set_seeds(seed)
        obj = get_objective(model, datasets, tuning_ranges)
        study = tune_objective(obj, study_name=model, num_samples=config_hyper["num_samples"], sampler=tune_sampler)
        best_params = study.best_trial.params
        # Save params
        path = hyper_path + model
        utils.save_yaml(path, best_params)
    print("Done tuning")

def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tune_sampler = RandomSampler(seed=seed)
    return tune_sampler


def get_objective(key, datasets, tuning_ranges):
    def obj(trial):
        gmsm = GMSM_binary(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"])
        #Sample params
        if key == "y" and datasets["d_train"].datatypes["y_type"] == "continuous":
            config = sample_params(trial, tuning_ranges["nf"])
        else:
            config = sample_params(trial, tuning_ranges["mlp"])
        # Fit
        if key in list(datasets["d_train"].data["m"].keys()):
            train_results = gmsm.fit_distribution_by_keys([key], config, d_train=datasets["d_train"], d_val=datasets["d_val"],
                                                          datatype="discrete")
        elif key == "y":
            train_results = gmsm.fit_distribution_by_keys([key], config, d_train=datasets["d_train"], d_val=datasets["d_val"],
                                                          datatype=datasets["d_train"].datatypes["y_type"])
        elif key == "y_regression":
            train_results = gmsm.fit_y_regression(config, d_train=datasets["d_train"], d_val=datasets["d_val"])
        elif key == "propensity":
            train_results = gmsm.fit_distribution_by_keys("a", config, d_train=datasets["d_train"], d_val=datasets["d_val"],
                                                          datatype=datasets["d_train"].datatypes["a_type"])
        else:
            raise ValueError("Invalid key in config: " + key)
        return train_results["val_results"][0]["val_obj"]

    return obj


def tune_objective(objective, study_name, num_samples=10, sampler=None):
    if sampler is not None:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=num_samples)

    print("Finished. Best trial:")
    trial_best = study.best_trial

    print("  Value: ", trial_best.value)

    print("  Params: ")
    for key, value in trial_best.params.items():
        print("    {}: {}".format(key, value))
    return study

def sample_params(trial, tuning_ranges):
    params = {}
    for param in tuning_ranges.keys():
        params[param] = trial.suggest_categorical(param, tuning_ranges[param])
    params["neptune"] = False
    return params
