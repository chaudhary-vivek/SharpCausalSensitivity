import utils.utils as utils
from data.data_generation import get_datasets
from models.gmsm_instantiations import GMSM_binary
from models.ensembles import SensitivityEnsemble
import torch
import random


def run_experiment(config, exp_function, end_function=None):
    seed = config["run"]["seed"]
    result_path = utils.get_project_path() + "/experiments/" + config["run"]["name"] + "/results/"
    results = []
    for i in range(config["run"]["n_runs"]):
        print("Starting run " + str(i+1) + " of " + str(config["run"]["n_runs"]))
        utils.set_seed(seed)
        seed = random.randint(0, 1000000)
        # Load data
        datasets = get_datasets(config["data"])
        # Train nuisance saved_models
        if config["run"]["train_models"]:
            nuisance = train_models(config["run"], datasets, run=i)
        else:
            nuisance = load_models(config["run"], datasets, run=i)
        # Run experiment, result should be dictionary of pandas dataframes
        result = exp_function(config["run"], datasets, nuisance)
        # Save results
        if "save" in config:
            if config["save"]:
                if result is not None:
                    for key in result.keys():
                        result[key].to_pickle(result_path + key + "_run_" + str(i) + ".pkl")
        results.append(result)
    if end_function is not None:
        end_function(config, results)
    print("Experiment finished")


def train_models(config_run, datasets, run=0):
    hyperpath = "/hyperparams/" + config_run["name"] + "/params/"
    savepath = "/experiments/" + config_run["name"] + "/saved_models/run_" + str(run) + "/"
    model_keys = config_run["model_keys"]
    #Create hyperparameter configuration
    model_config = {}
    for key in model_keys:
        model_config[key] = utils.load_yaml(hyperpath + key)
        model_config[key]["neptune"] = config_run["validation"]

    if config_run["n_bootstrap"] == 0:
        print("Training nuisance models for run " + str(run))
        gmsm_nuisance = GMSM_binary(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"])
        gmsm_nuisance.fit(model_config, d_train=datasets["d_train"], d_val=datasets["d_val"], y_scale=config_run["scale_y"])
        trained_models = gmsm_nuisance.models
        # Save saved_models
        for key in model_keys:
            utils.save_pytorch_model(savepath + "full_data/" + key, trained_models[key])
        #Save scaling params
        save_scale_params(gmsm_nuisance, savepath + "full_data/scale_params_y")
        return {"models": trained_models, "scaling_params": gmsm_nuisance.scaling_params}
    elif config_run["n_bootstrap"] > 0:
        print("Training nuisance ensemble for run " + str(run))
        gmsm_ens_nuisance = SensitivityEnsemble(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"],
                                                model_class=GMSM_binary)
        gmsm_ens_nuisance.fit_ensemble(model_config, d_train=datasets["d_train"], d_val=datasets["d_val"], y_scale=config_run["scale_y"],
                                       k=config_run["n_bootstrap"], resample=config_run["resample"])
        trained_ensemble = [model.models for model in gmsm_ens_nuisance.ensemble]
        # Save models
        for i in range(len(trained_ensemble)):
            for key in model_keys:
                utils.save_pytorch_model(savepath + "bootstrapped/" + key + "_" + str(i), trained_ensemble[i][key])
        #Save scaling params
        save_scale_params(gmsm_ens_nuisance, savepath + "bootstrapped/scale_params_y")
        return {"models": trained_ensemble, "scaling_params": gmsm_ens_nuisance.scaling_params}
    else:
        raise ValueError("Invalid value for n_bootstrap: " + str(config_run["n_bootstrap"]))

def load_models(config_run, datasets, run=0):
    hyperpath = "/hyperparams/" + config_run["name"] + "/params/"
    savepath = "/experiments/" + config_run["name"] + "/saved_models/run_" + str(run) + "/"

    model_keys = config_run["model_keys"]
    #Create hyperparameter configuration
    model_config = {}
    for key in model_keys:
        model_config[key] = utils.load_yaml(hyperpath + key)
        model_config[key]["neptune"] = False
        #Set epochs to 0 to avoid training
        model_config[key]["epochs"] = 0

    if config_run["n_bootstrap"] == 0:
        # Initialize models with hyperparameters
        gmsm_init = GMSM_binary(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"])
        gmsm_init.fit(model_config, d_train=datasets["d_train"], d_val=datasets["d_val"],
                          y_scale=False)
        init_models = gmsm_init.models
        # Load saved_models
        for key in model_keys:
            utils.load_pytorch_model(savepath + "full_data/" + key, init_models[key])
        scaling_params = utils.load_yaml(savepath + "full_data/scale_params_y")
        return {"models": init_models, "scaling_params": scaling_params}
    elif config_run["n_bootstrap"] > 0:
        gmsm_ens_init = SensitivityEnsemble(datasets["causal_graph"],
                                                y_type=datasets["d_train"].datatypes["y_type"],
                                                model_class=GMSM_binary)
        gmsm_ens_init.fit_ensemble(model_config, d_train=datasets["d_train"], d_val=datasets["d_val"],
                                       y_scale=config_run["scale_y"],
                                       k=config_run["n_bootstrap"])
        init_ensemble = gmsm_ens_init.ensemble
        for i in range(len(init_ensemble)):
            for key in model_keys:
                utils.load_pytorch_model(savepath + "bootstrapped/" + key + "_" + str(i), init_ensemble[i].models[key])
        scaling_params = utils.load_yaml(savepath + "bootstrapped/scale_params_y")
        return {"models": [model.models for model in init_ensemble], "scaling_params": scaling_params}
    else:
        raise ValueError("Invalid value for n_bootstrap: " + str(config_run["n_bootstrap"]))


def save_scale_params(gmsm, savepath):
    scaling_params = gmsm.scaling_params.copy()
    if torch.is_tensor(scaling_params["mean"]):
        scaling_params["mean"] = scaling_params["mean"].item()
        scaling_params["sd"] = scaling_params["sd"].item()
    utils.save_yaml(savepath, scaling_params)