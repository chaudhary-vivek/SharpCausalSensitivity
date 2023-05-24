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
    # Create plots for paper (oracle gamma + bounds)
    plotting.plot_gamma_scm(scm, a_list=[0.6], binary=False, path_rel="/experiments/exp_sim_continuous/results/plot_gamma_continuous_0.6.pdf")
    plotting.plot_bounds_scm_quantile(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0.6], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_continuous/results/plot_exp_continuous_0.6_q0.3.pdf", q=0.3)
    plotting.plot_bounds_scm_quantile(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0.6], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_continuous/results/plot_exp_continuous_0.6_q0.5.pdf", q=0.5)
    plotting.plot_bounds_scm_quantile(gmsm, scm, gamma_dict, n_samples=20000, n_samples_oracle=80000, a_int=[0.6], bootstrap=bootstrapping,
                                path_rel="/experiments/exp_sim_continuous/results/plot_exp_continuous_0.6_q0.7.pdf", q=0.7)

    return None

if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_sim_continuous/config")
    run_experiment(config_run, exp_function=exp_function)

