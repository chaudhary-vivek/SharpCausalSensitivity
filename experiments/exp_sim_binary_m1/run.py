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
    if config_run["n_bootstrap"] == 0:
        gmsm = GMSM_binary(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], models=nuisance["models"],
                           scaling_params=nuisance["scaling_params"])
    else:
        gmsm = SensitivityEnsemble(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], model_list=nuisance["models"],
                           scaling_params=nuisance["scaling_params"], model_class=GMSM_binary)
    # Data generating SCM
    scm = datasets["scm"]
    gamma_dict = {"m1": torch.tensor([1, 2, 2, 2]), "y": torch.tensor([1, 1, 1.3, 1.7])}
    #Plotting ground truth
    if config_run["plotting"] and not bootstrapping:
        # Propensity score fit
        val_plots.plot_binary_dist_fit(gmsm, scm)

        # Mediator density fit
        val_plots.plot_binary_dist_fit(gmsm, scm, key="m1", a_cond=1)

        # Outcome density fit (for single x)
        val_plots.plot_density_fit(x_cond=-0.9, a_cond=0, m1_cond=0, gsm=gmsm, scm=scm, key="y", n_samples=10000,
                                   grid_size=2000, bins=40)
        val_plots.plot_bounds(gmsm, scm, gamma_dict, n_samples=10000, a_int=[1, 0],bootstrap=bootstrapping)

    # Create plots for paper (oracle gamma, propensity scores + bounds)
    plotting.plot_gamma_scm(scm, a_list=[1, 0], binary=True,
                            path_rel="/experiments/exp_sim_binary_m1/results/plot_gamma_binary_m1.pdf")
    plotting.plot_bounds_scm(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[1, 0], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_binary_m1/results/plot_exp_binary_m1.pdf")
    plotting.plot_bounds_scm(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0, 1], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_binary_m1/results/plot_exp_binary_m1_0_1.pdf")

    return None

if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_sim_binary_m1/config")
    run_experiment(config_run, exp_function=exp_function)

