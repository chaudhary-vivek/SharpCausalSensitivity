import utils.utils as utils
from experiments.main import run_experiment
import utils.plotting as plotting
from models.gmsm_instantiations import GMSM_continuous, GMSM_weighted
from models.cmsm import CMSM
from models.ensembles import SensitivityEnsemble
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def exp_function(config_run, datasets, nuisance):
    #Create GMSM from nuisance models
    models_gmsm = nuisance["models"].copy()
    models_gmsm.pop("y_regression")
    gmsm = GMSM_continuous(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], models=models_gmsm,
                       scaling_params=nuisance["scaling_params"])

    #Define weighted GMSM (weight function)
    def wf(data):
        return torch.where(data.data["x"] > 0, 1, 0)
    gmsm_weighted = GMSM_weighted(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], models=models_gmsm,
                       scaling_params=nuisance["scaling_params"], weight_function=wf)
    cmsm = CMSM(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"], models=nuisance["models"],
                       scaling_params=nuisance["scaling_params"])
    # Data generating SCM
    scm = datasets["scm"]
    # Compute bounds
    gamma_dict = {"y": torch.tensor([1.2, 1.5, 2])}
    x_test = np.random.uniform(-1, 1, (500, 1))
    result_dict = {}
    a_int = [0.6]
    for a in a_int:
        start_time = time.time()
        bounds_gmsm = gmsm.compute_bounds_average(x_test, [a], gamma_dict, n_samples=5000)
        gmsm_time = time.time() - start_time
        bounds_gmsm_plus = bounds_gmsm["Q_plus"].detach().numpy()
        bounds_gmsm_minus = bounds_gmsm["Q_minus"].detach().numpy()

        start_time = time.time()
        bounds_gmsmw = gmsm_weighted.compute_bounds_average(x_test, [a], gamma_dict, n_samples=5000)
        gmsmw_time = time.time() - start_time
        bounds_gmsmw_plus = bounds_gmsmw["Q_plus"].detach().numpy()
        bounds_gmsmw_minus = bounds_gmsmw["Q_minus"].detach().numpy()

        start_time = time.time()
        bounds_cmsm = cmsm.compute_bounds_average(x_test, [a], gamma_dict, n_samples=5000)
        cmsm_time = time.time() - start_time
        bounds_cmsm_plus = bounds_cmsm["Q_plus"].detach().numpy()
        bounds_cmsm_minus = bounds_cmsm["Q_minus"].detach().numpy()

        # Size of bounds
        gmsm_diff = bounds_gmsm_plus - bounds_gmsm_minus
        gmsmw_diff = bounds_gmsmw_plus - bounds_gmsmw_minus
        cmsm_diff = bounds_cmsm_plus - bounds_cmsm_minus

        #Oracle coverage
        cate = np.mean(scm.get_true_effect(x_test, n_samples=20000, a1_int=a_int))
        gmsm_cov = np.minimum((bounds_gmsm_plus > cate).astype(int), (cate > bounds_gmsm_minus).astype(int))
        gmsmw_cov = np.minimum((bounds_gmsmw_plus > cate).astype(int), (cate > bounds_gmsmw_minus).astype(int))
        cmsm_cov = np.minimum((bounds_cmsm_plus > cate).astype(int), (cate > bounds_cmsm_minus).astype(int))


        result_dict = result_dict | {"gmsm_diff_" + str(a): gmsm_diff, "gmsmw_diff_"+ str(a): gmsmw_diff, "cmsm_diff_"+ str(a): cmsm_diff,
            "gmsm_cov_"+ str(a): gmsm_cov, "gmsmw_cov_"+ str(a): gmsmw_cov, "cmsm_cov_"+ str(a): cmsm_cov,
            "gmsm_time_"+ str(a): gmsm_time, "gmsmw_time_"+ str(a): gmsmw_time, "cmsm_time_"+ str(a): cmsm_time}
    return result_dict

# Function executed at the end of the experiment
def end_function(config, results):
    #aggregate results
    keys = results[0].keys()

    # Create an empty dictionary to store the averaged arrays
    results_avg = {}
    results_std = {}

    # Iterate over the keys
    for key in keys:
        # Get the arrays for the current key from all dictionaries in the list
        arrays = [d[key] for d in results]

        # Compute the average of the arrays along the first axis
        averaged_array = np.mean(arrays, axis=0)
        std_array = np.std(arrays, axis=0)

        # Store the averaged array in the new dictionary
        results_avg[key] = averaged_array
        results_std[key] = std_array

    print("Means")
    print(results_avg)
    print("Std")
    print(results_std)



if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_sim_continuous_weight/config")
    run_experiment(config_run, exp_function=exp_function, end_function=end_function)

