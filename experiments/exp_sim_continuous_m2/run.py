import utils.utils as utils
from experiments.main import run_experiment
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
        gmsm = GMSM_continuous(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], models=nuisance["models"],#[0].models,
                           scaling_params=nuisance["scaling_params"])
    else:
        gmsm = SensitivityEnsemble(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], model_list=nuisance["models"],
                           scaling_params=nuisance["scaling_params"], model_class=GMSM_continuous)
    # Data generating SCM
    scm = datasets["scm"]
    # Compute bounds for indirect effect
    gamma_dict = {"m1": torch.tensor([1, 1.2, 1.2, 1.2]), "m2": torch.tensor([1, 1, 1.2, 1.2]),
                  "y": torch.tensor([1, 1, 1, 1.1])}

    # Create plots for paper (oracle gamma + bounds)
    plotting.plot_gamma_scm(scm, a_list=[0.2, 0.4, 0.5], binary=False, path_rel="/experiments/exp_sim_continuous_m2/results/plot_gamma_continuous_m2_0.2_0.4_0.5.pdf")
    plotting.plot_bounds_scm(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0.2, 0.4, 0.5], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_continuous_m2/results/plot_exp_continuous_m2_0.2_0.4_0.5.pdf")
    plotting.plot_gamma_scm(scm, a_list=[0.4, 0.5, 0.3], binary=False, path_rel="/experiments/exp_sim_continuous_m2/results/plot_gamma_continuous_m2_0.4_0.5_0.3.pdf")
    plotting.plot_bounds_scm(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0.6, 0.5, 0.4], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_continuous_m2/results/plot_exp_continuous_m2_0.4_0.5_0.3.pdf")

    return None

if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_sim_continuous_m2/config")
    run_experiment(config_run, exp_function=exp_function)

