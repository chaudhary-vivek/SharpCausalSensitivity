import utils.utils as utils
from experiments.main import run_experiment
import utils.validation_plots as val_plots
import utils.plotting as plotting
from models.gmsm_instantiations import GMSM_continuous
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
        gmsm = GMSM_continuous(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], models=nuisance["models"],#[0],
                           scaling_params=nuisance["scaling_params"])
    else:
        gmsm = SensitivityEnsemble(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], model_list=nuisance["models"],
                           scaling_params=nuisance["scaling_params"], model_class=GMSM_continuous)
    # Data generating SCM
    scm = datasets["scm"]
    # Compute bounds
    gamma_dict = {"y": torch.tensor([1, 1.1, 1.2, 1.5])}
    #Plotting ground truth
    if config_run["plotting"] and not bootstrapping:
        # Outcome density fit (for single x)
        val_plots.plot_density_fit(x_cond=-0.5, a_cond=0.6, gsm=gmsm, scm=scm, key="y", n_samples=10000,
                                   grid_size=2000, bins=40, support_left=-5, support_right=5, a_type="continuous")
        val_plots.plot_bounds(gmsm, scm, gamma_dict, n_samples=10000, a_int=[0.6], bootstrap=bootstrapping, plot_cond=True, a_type="continuous")
    # Create plots for paper (oracle gamma + bounds)
    plotting.plot_gamma_scm(scm, a_list=[0.6], binary=False, path_rel="/experiments/exp_sim_continuous/results/plot_gamma_continuous_0.6.pdf")
    plotting.plot_bounds_scm(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0.6], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_continuous/results/plot_exp_continuous_0.6.pdf")

    plotting.plot_gamma_scm(scm, a_list=[0.9], binary=False, path_rel="/experiments/exp_sim_continuous/results/plot_gamma_continuous_0.9.pdf")
    plotting.plot_bounds_scm(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0.9], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_continuous/results/plot_exp_continuous_0.9.pdf")
    return None

if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_sim_continuous/config")
    run_experiment(config_run, exp_function=exp_function)

