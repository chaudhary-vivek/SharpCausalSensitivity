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
    # Create GMSM from nuisance models
    if config_run["n_bootstrap"] == 0:
        gmsm = GMSM_binary(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"],
                           models=nuisance["models"],
                           scaling_params=nuisance["scaling_params"])
    else:
        gmsm = SensitivityEnsemble(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"],
                                   model_list=nuisance["models"],
                                   scaling_params=nuisance["scaling_params"], model_class=GMSM_binary)

    gamma_dict = {"m1": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), "y": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
    x = datasets["d_train"].data["x"].detach().numpy()
    bootstrap = config_run["n_bootstrap"] > 0
    plotting.plot_bounds_real(gmsm, gamma_dict, a_int1=[0, 1], a_int2=[0, 0], x_test=x, n_samples=10000,
                              q=None, bootstrap=bootstrap, path_rel="/experiments/exp_real/results/plot_real.pdf")
    return None

if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_real/config")
    run_experiment(config_run, exp_function=exp_function)

