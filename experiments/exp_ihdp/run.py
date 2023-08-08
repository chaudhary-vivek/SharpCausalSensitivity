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
import seaborn as sns

def exp_function(config_run, datasets, nuisance):
    # Create GMSM from nuisance models
    gmsm = GMSM_binary(datasets["causal_graph"], y_type=datasets["d_train"].datatypes["y_type"],
                       models=nuisance["models"],
                       scaling_params=nuisance["scaling_params"])
    d_test = datasets["d_test"]
    x_test = d_test.data["x"].detach().numpy()
    n_test = x_test.shape[0]
    # Check for overlap violations
    prop = gmsm.predict_by_key("propensity", d_test)
    # bool tensor that contains true if prop < 0.05 or prop > 0.95
    violated = torch.logical_or(prop < 0.05, prop > 0.95)
    # Exclude points with overlap violations from evaluation
    #d_test.data["x"] = d_test.data["x"][~violated.squeeze()]
    #d_test.data["a"] = d_test.data["a"][~violated.squeeze()]
    #d_test.data["y"] = d_test.data["y"][~violated.squeeze()]
    # number of overlap violations
    num_violations = torch.sum(violated)
    print("Overlap violations: ", num_violations)
    if num_violations > (0.3 * n_test):
        print("Overlap violations: ", num_violations)
        print("Exclude from evaluation")
        return None

    gamma_dict = {"y": torch.tensor(np.linspace(1, 20, 100), dtype=torch.float32)}
    deferral_rates = np.linspace(0, 1, 11)

    # Compute bounds (shape is (n_test, n_gamma))
    n_gammas = len(gamma_dict["y"])
    #Mean
    bounds1_mean = gmsm.compute_bounds(x_test, [1], gamma_dict, n_samples=1000)
    bounds1_mean_plus = bounds1_mean["Q_plus"].detach().numpy()
    bounds1_mean_minus = bounds1_mean["Q_minus"].detach().numpy()
    bounds0_mean = gmsm.compute_bounds(x_test, [0], gamma_dict, n_samples=1000)
    bounds0_mean_plus = bounds0_mean["Q_plus"].detach().numpy()
    bounds0_mean_minus = bounds0_mean["Q_minus"].detach().numpy()
    #Quantile 1
    bounds1_q = gmsm.compute_bounds(x_test, [1], gamma_dict, n_samples=1000, q=0.4)
    bounds1_q_plus = bounds1_q["Q_plus"].detach().numpy()
    bounds1_q_minus = bounds1_q["Q_minus"].detach().numpy()
    bounds0_q = gmsm.compute_bounds(x_test, [0], gamma_dict, n_samples=1000, q=0.6)
    bounds0_q_plus = bounds0_q["Q_plus"].detach().numpy()
    bounds0_q_minus = bounds0_q["Q_minus"].detach().numpy()
    #Quantile 2
    bounds1_q2 = gmsm.compute_bounds(x_test, [1], gamma_dict, n_samples=1000, q=0.2)
    bounds1_q2_plus = bounds1_q2["Q_plus"].detach().numpy()
    bounds1_q2_minus = bounds1_q2["Q_minus"].detach().numpy()
    bounds0_q2 = gmsm.compute_bounds(x_test, [0], gamma_dict, n_samples=1000, q=0.8)
    bounds0_q2_plus = bounds0_q2["Q_plus"].detach().numpy()
    bounds0_q2_minus = bounds0_q2["Q_minus"].detach().numpy()

    # Decisions of associated policies (ignoring uncertainty)
    decisions_mean = np.where(bounds1_mean_plus[:, 0] > bounds0_mean_plus[:, 0], 1, 0)
    decisions_q = np.where(bounds1_q_plus[:, 0] > bounds0_q_plus[:, 0], 1, 0)
    decisions_q2 = np.where(bounds1_q2_plus[:, 0] > bounds0_q2_plus[:, 0], 1, 0)
    num_treated_mean = np.sum(decisions_mean)
    num_treated_q = np.sum(decisions_q)
    num_treated_q2 = np.sum(decisions_q2)
    # Ground truth decisions
    decisions_gt = np.where(datasets["y1_test"] > datasets["y0_test"], 1, 0).squeeze()

    # For each x, compute minimum gamma for which treated/ untreated bounds are overlapping
    # (shape is (n_test, 1))
    gammas = gamma_dict["y"].detach().numpy()
    min_gamma_mean = np.zeros((x_test.shape[0]))
    min_gamma_q = np.zeros((x_test.shape[0]))
    min_gamma_q2 = np.zeros((x_test.shape[0]))
    for i in range(n_test):
        overlap_onebiggerzero_mean =np.logical_and(bounds1_mean_minus[i, :] < bounds0_mean_plus[i, :],
                                    np.repeat(bounds1_mean_minus[i, 0] > bounds0_mean_minus[i, 0], n_gammas))
        overlap_zerobiggerone_mean = np.logical_and(bounds1_mean_plus[i, :] > bounds0_mean_minus[i, :],
                                    np.repeat(bounds1_mean_minus[i, 0] < bounds0_mean_minus[i, 0], n_gammas))
        overlap_mean = np.logical_or(overlap_onebiggerzero_mean, overlap_zerobiggerone_mean)
        gammas_overlap_mean = gammas[overlap_mean]
        if len(gammas_overlap_mean) > 0:
            min_gamma_mean[i] = np.min(gammas_overlap_mean)
        else:
            min_gamma_mean[i] = gammas[-1]
        overlap_onebiggerzero_q =np.logical_and(bounds1_q_minus[i, :] < bounds0_q_plus[i, :],
                                    np.repeat(bounds1_q_minus[i, 0] > bounds0_q_minus[i, 0], n_gammas))
        overlap_zerobiggerone_q = np.logical_and(bounds1_q_plus[i, :] > bounds0_q_minus[i, :],
                                    np.repeat(bounds1_q_minus[i, 0] < bounds0_q_minus[i, 0], n_gammas))
        overlap_q = np.logical_or(overlap_onebiggerzero_q, overlap_zerobiggerone_q)
        gammas_overlap_q = gammas[overlap_q]
        if len(gammas_overlap_q) > 0:
            min_gamma_q[i] = np.min(gammas_overlap_q)
        else:
            min_gamma_q[i] = gammas[-1]

        overlap_onebiggerzero_q2 =np.logical_and(bounds1_q2_minus[i, :] < bounds0_q2_plus[i, :],
                                    np.repeat(bounds1_q2_minus[i, 0] > bounds0_q2_minus[i, 0], n_gammas))
        overlap_zerobiggerone_q2 = np.logical_and(bounds1_q2_plus[i, :] > bounds0_q2_minus[i, :],
                                    np.repeat(bounds1_q2_minus[i, 0] < bounds0_q2_minus[i, 0], n_gammas))
        overlap_q2 = np.logical_or(overlap_onebiggerzero_q2, overlap_zerobiggerone_q2)
        gammas_overlap_q2 = gammas[overlap_q2]
        if len(gammas_overlap_q2) > 0:
            min_gamma_q2[i] = np.min(gammas_overlap_q2)
        else:
            min_gamma_q2[i] = gammas[-1]

    # Get the indices of sorted arrays by gammas
    idx_mean = min_gamma_mean.argsort()
    idx_q = min_gamma_q.argsort()
    idx_q2 = min_gamma_q2.argsort()
    # Defer samples with highest min_gamma
    err_rates_mean = []
    err_rates_q = []
    err_rates_q2 = []
    for i, rate in enumerate(deferral_rates):
        # Number of samples to defer
        n_defer = int(np.floor(rate * n_test))
        if n_defer < n_test:
            idx_keep_mean = idx_mean[n_defer:]
            idx_keep_q = idx_q[n_defer:]
            idx_keep_q2 = idx_q2[n_defer:]
            keep_mean = decisions_mean[idx_keep_mean]
            keep_q = decisions_q[idx_keep_q]
            keep_q2 = decisions_q2[idx_keep_q2]
            keep_gt_mean = decisions_gt[idx_keep_mean]
            keep_gt_q = decisions_gt[idx_keep_q]
            keep_gt_q2 = decisions_gt[idx_keep_q2]
            # Error rates of deferral policies
            #err_rates_mean.append(np.mean(np.abs(keep_mean - keep_gt)))
            #err_rates_q.append(np.mean(np.abs(keep_q - keep_gt)))
            # Weighted error rates of deferral policies
            bad_errors_mean = np.where(keep_mean > keep_gt_mean, 20, 0)
            ok_errors_mean = np.where(keep_mean < keep_gt_mean, 1, 0)
            err_rates_mean.append(np.sum(bad_errors_mean + ok_errors_mean) / (n_test - n_defer))
            bad_errors_q = np.where(keep_q > keep_gt_q, 20, 0)
            ok_errors_q = np.where(keep_q < keep_gt_q, 1, 0)
            err_rates_q.append(np.sum(bad_errors_q + ok_errors_q) / (n_test - n_defer))
            bad_errors_q2 = np.where(keep_q2 > keep_gt_q2, 20, 0)
            ok_errors_q2 = np.where(keep_q2 < keep_gt_q2, 1, 0)
            err_rates_q2.append(np.sum(bad_errors_q2 + ok_errors_q2) / (n_test - n_defer))

            num_bad_mean = np.sum(bad_errors_mean)
            num_bad_q = np.sum(bad_errors_q)
            num_ok_mean = np.sum(ok_errors_mean)
            num_ok_q = np.sum(ok_errors_q)
            pass
        else:
            err_rates_mean.append(0)
            err_rates_q.append(0)
            err_rates_q2.append(0)


    return {"err_rates_mean": err_rates_mean, "err_rates_q": err_rates_q, "err_rates_q2": err_rates_q2}

# Function executed at the end of the experiment
def end_function(config, results):
    #aggregate results
    err_rates_mean = [result["err_rates_mean"] for result in results if result is not None]
    err_rates_q = [result["err_rates_q"] for result in results if result is not None]
    err_rates_q2 = [result["err_rates_q2"] for result in results if result is not None]
    results_avg = {"err_rates_mean": np.mean(err_rates_mean, axis=0),
                   "err_rates_q": np.mean(err_rates_q, axis=0),
                   "err_rates_q2": np.mean(err_rates_q2, axis=0)}
    results_std = {"err_rates_mean": np.std(err_rates_mean, axis=0),
                   "err_rates_q": np.std(err_rates_q, axis=0),
                   "err_rates_q2": np.std(err_rates_q2, axis=0)}

    print("Means")
    print(results_avg)
    print("Std")
    print(results_std)

    # Plot results average error rates over deferral rates
    colors = ["darkblue", "darkgreen", "darkred"]
    deferral_rates = np.linspace(0, 1, 11)
    sns.set(font_scale=1.15, style="whitegrid")
    fig, ax = plt.subplots()
    ax.plot(deferral_rates, results_avg["err_rates_mean"], label="Expectation policy", color=colors[0])
    ax.plot(deferral_rates, results_avg["err_rates_q"], label=r"Quantile policy $q=0.4$", color=colors[1])
    ax.plot(deferral_rates, results_avg["err_rates_q2"], label=r"Quantile policy $q=0.2$", color=colors[2])
    # Plot error bars using standard deviations
    ax.fill_between(deferral_rates, results_avg["err_rates_mean"] - results_std["err_rates_mean"],
                    results_avg["err_rates_mean"] + results_std["err_rates_mean"], alpha=0.2, color=colors[0])
    ax.fill_between(deferral_rates, results_avg["err_rates_q"] - results_std["err_rates_q"],
                    results_avg["err_rates_q"] + results_std["err_rates_q"], alpha=0.2, color=colors[1])
    ax.fill_between(deferral_rates, results_avg["err_rates_q2"] - results_std["err_rates_q2"],
                    results_avg["err_rates_q2"] + results_std["err_rates_q2"], alpha=0.2, color=colors[2])
    ax.set_xlabel("Deferral rate")
    ax.set_ylabel("Weighted error rate")
    ax.legend()
    plt.savefig(utils.get_project_path() + "/experiments/exp_ihdp/results/plot_ihdp.pdf")
    plt.show()




if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/exp_ihdp/config")
    run_experiment(config_run, exp_function=exp_function, end_function=end_function)

