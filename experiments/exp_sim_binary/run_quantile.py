import utils.utils as utils
from experiments.main import run_experiment
import utils.validation_plots as val_plots
import utils.plotting as plotting
from models.gmsm_instantiations import GMSM_binary
from models.ensembles import SensitivityEnsemble
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exp_function(config_run, datasets, nuisance):
    bootstrapping = config_run["n_bootstrap"] > 0
    #bootstrapping = False
    #Create GMSM from nuisance models
    if not bootstrapping:
        gmsm = GMSM_binary(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], models=nuisance["models"],#[0].models,
                           scaling_params=nuisance["scaling_params"])
    else:
        gmsm = SensitivityEnsemble(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], model_list=nuisance["models"],
                           scaling_params=nuisance["scaling_params"], model_class=GMSM_binary)
    # Data generating SCM
    scm = datasets["scm"]
    gamma_dict = {"y": torch.tensor([1, 1.3, 1.6, 2])}

    # Create plots for paper (oracle gamma, propensity scores + bounds)
    plotting.plot_gamma_scm(scm, a_list=[1], binary=True, path_rel="/experiments/exp_sim_binary/results/plot_gamma_binary.pdf")
    plotting.plot_bounds_scm_quantile(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[1], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_binary/results/plot_exp_binary_q0.3.pdf", q=0.3, legend_loc="upper right")
    plotting.plot_bounds_scm_quantile(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[1], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_binary/results/plot_exp_binary_q0.5.pdf", q=0.5, legend_loc="upper right")
    plotting.plot_bounds_scm_quantile(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[1], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_binary/results/plot_exp_binary_q0.7.pdf", q=0.7, legend_loc="upper right")
    return None

if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_sim_binary/config")
    run_experiment(config_run, exp_function=exp_function)

