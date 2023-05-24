import matplotlib.pyplot as plt
from data.data_structures import GSM_Dataset
import numpy as np
import torch
import seaborn as sns
import pandas as pd


# Distribution fit (for binary treatment)
def plot_binary_dist_fit(gsm, scm_binary, x_test=None, a_cond=1, m1_cond=1, key="propensity", n_samples=5000):
    if x_test is None:
        x_test = np.expand_dims(np.linspace(-1, 1, 500), axis=1)
    d_predict = GSM_Dataset(x=x_test, a=np.full_like(x_test, a_cond), m={"m1": np.full_like(x_test, m1_cond)}, y=None)
    # Predict distribution
    predictions = gsm.predict_by_key(key, d_predict)
    # Observed distribution
    if key == "propensity":
        observed = scm_binary.propensity_obs(a=1, x=x_test)
    else:
        if key == "m1":
            observed = scm_binary.get_conditional_dist(x_test, n_samples, a_cond=a_cond, key="m1")
        elif key == "m2":
            observed = scm_binary.get_conditional_dist(x_test, n_samples, a_cond=a_cond, m1_cond=m1_cond, key="m2")
        else:
            raise ValueError("Key not recognized")
    # Plot propensity predictions
    plt.plot(x_test, predictions.detach().numpy(), label="Predicted " + key)
    plt.plot(x_test, observed, label="Observed " + key)
    plt.title("Predicted vs observed binary distribution")
    plt.legend()
    plt.show()


# Density fit (either for conditional outcome or treatment density)
def plot_density_fit(x_cond, gsm, scm, key="y", a_cond=None, m1_cond=None, m2_cond=None, n_samples=10000,
                     grid_size=2000, bins=35, a_type="binary", tol_a=0.01, support_left=-4, support_right=4):
    x_test = np.full((10000, 1), x_cond)
    # Obtaining observed conditional outcome distribution
    [_, a_test, m1_test, m2_test, y_test] = scm.generate_data(n_samples=n_samples, int_x=x_test)
    if key == "y":
        if a_type == "binary":
            idx_a = a_test[:, 0] == a_cond
        else:
            idx_a = np.abs(a_test[:, 0] - a_cond) < tol_a
        if m1_cond is not None:
            idx_m1 = m1_test[:, 0] == m1_cond
            if m2_cond is not None:
                idx_m2 = m2_test[:, 0] == m2_cond
                d_observed = y_test[idx_a & idx_m1 & idx_m2, :]
                data_predict = GSM_Dataset(x=np.full((1, 1), x_cond), a=np.full((1, 1), a_cond),
                                           m={"m1": np.full((1, 1), m1_cond),
                                              "m2": np.full((1, 1), m2_cond)}, y=None)
            else:
                d_observed = y_test[idx_a & idx_m1, :]
                data_predict = GSM_Dataset(x=np.full((1, 1), x_cond), a=np.full((1, 1), a_cond),
                                           m={"m1": np.full((1, 1), m1_cond)},
                                           y=None)
        else:
            d_observed = y_test[idx_a, :]
            data_predict = GSM_Dataset(x=np.full((1, 1), x_cond), a=np.full((1, 1), a_cond), m=None,
                                       y=None)
    elif key == "a":
        d_observed = a_test
        data_predict = GSM_Dataset(x=np.full((1, 1), x_cond), a=None, m=None, y=None)
    else:
        raise Exception("Key not supported")

    # Predict density
    #ygrid = torch.reshape(torch.linspace(np.min(d_observed), np.max(d_observed), grid_size),
    #                      (grid_size, 1))
    ygrid = torch.reshape(torch.linspace(support_left, support_right, grid_size),(grid_size, 1))
    d_predict = gsm.predict_by_key(key, data_predict, y_grid=ygrid).detach().numpy()
    # Plot
    plt.hist(d_observed, density=True, bins=bins, label="Observed Density")
    plt.plot(ygrid.detach().numpy(), d_predict, label="Predicted Density")
    plt.legend()
    plt.show()


def plot_bounds(gsm, scm, gamma_dict, x_test=None, a_int=None, a_int2=None, n_samples=1000, a_type="binary", tol_a=0.05, q=None, bootstrap=False,
                plot_cond=True):
    # Compute bounds
    if x_test is None:
        x_test = np.expand_dims(np.linspace(-1, 1, 200), axis=1)
    if a_int is None:
        if "m2" in gsm.causal_graph["nodes"]:
            a_int = [1, 1, 1]
        elif "m1" in gsm.causal_graph["nodes"]:
            a_int = [1, 1]
        else:
            a_int = [1]
    if bootstrap:
        if a_int2 is None:
            bounds = gsm.compute_bounds_ensemble(x_test, gamma_dict, a_int1=a_int, n_samples=n_samples, alpha=0.05, q=q, average=False)
        else:
            bounds = gsm.compute_bounds_ensemble(x_test, gamma_dict, a_int1=a_int, a_int2=a_int2, n_samples=n_samples, alpha=0.05, q=q, average=False)
        bounds_plus = bounds["Q_plus"]["mean"].detach().numpy()
        bounds_minus = bounds["Q_minus"]["mean"].detach().numpy()
    else:
        if a_int2 is None:
            bounds = gsm.compute_bounds(x_test, a_int, gamma_dict, n_samples=n_samples, q=q)
        else:
            bounds = gsm.compute_bounds_difference(x_test, gamma_dict, n_samples=n_samples, a_int1=a_int, a_int2=a_int2, q=None)
        bounds_plus = bounds["Q_plus"].detach().numpy()
        bounds_minus = bounds["Q_minus"].detach().numpy()

    # True and conditional effects
    if "m2" in gsm.causal_graph["nodes"]:
        y_cond = scm.get_conditional_effect(x_test, n_samples=40000, a1_cond=a_int[0], a2_cond=a_int[1], a3_cond=a_int[2], a_type=a_type, tol_a=tol_a)
        y_int = scm.get_true_effect(x_test, n_samples=40000, a1_int=a_int[0], a2_int=a_int[1], a3_int=a_int[2])
    elif "m1" in gsm.causal_graph["nodes"]:
        y_cond = scm.get_conditional_effect(x_test, n_samples=20000, a1_cond=a_int[0], a2_cond=a_int[1], a_type=a_type, tol_a=tol_a)
        y_int = scm.get_true_effect(x_test, n_samples=20000, a1_int=a_int[0], a2_int=a_int[1])
    else:
        y_cond = scm.get_conditional_effect(x_test, n_samples=10000, a1_cond=a_int[0], a_type=a_type, tol_a=tol_a)
        y_int = scm.get_true_effect(x_test, n_samples=40000, a1_int=a_int[0])

    # Plot
    data_plot = pd.DataFrame({"x_test": x_test[:, 0], "y_cond": y_cond[:, 0], "y_int": y_int[:, 0]})
    # Create a Seaborn plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    # plt.plot(x_test, y_cond, label="Conditional")
    if plot_cond:
        sns.lineplot(data=data_plot, x="x_test", y="y_cond", label="Conditional", color="darkblue", linewidth=2)
    sns.lineplot(data=data_plot, x="x_test", y="y_int", label="Oracle", color="darkgreen", linewidth=2)

    num_gammas = len(gamma_dict[list(gamma_dict.keys())[0]])
    colors = sns.color_palette("flare", num_gammas)
    legend_labels = {"y": "$\Gamma_Y$", "m1": "$\Gamma_{M_1}$", "m2": "$\Gamma_{M_2}$"}
    for i in range(num_gammas):
        keys = gamma_dict.keys()
        # Create label
        label = ""
        for key in keys:
            label += legend_labels[key] + "=" + str(gamma_dict[key][i].detach().numpy()) + ", "
        if i == 0:
            plt.fill_between(x_test[:, 0], bounds_plus[:, 0], bounds_minus[:, 0], alpha=0.5, label=label, color=colors[i])
        else:
            # Fill between two times: bounds_plus[:, i-1] and bounds_plus[:, i], and bounds_minus[:, i-1] and bounds_minus[:, i]
            # Use the same label and color for both
            plt.fill_between(x_test[:, 0], bounds_plus[:, i - 1], bounds_plus[:, i], alpha=0.5, label=label, color=colors[i])
            plt.fill_between(x_test[:, 0], bounds_minus[:, i - 1], bounds_minus[:, i], alpha=0.5, color=colors[i])
    plt.legend()
    plt.show()


def plot_oracle_confounding(scm, keys=None, x_test=None, a=1):
    if x_test is None:
        x_test = np.expand_dims(np.linspace(-1, 1, 200), axis=1)
    if keys is None:
        keys = ["m1", "m2", "y"]
    prop_obs = scm.propensity_obs(a=a, x=x_test)
    #plt.plot(x_test, prop_obs, label="Observed propensity score")
    for key in keys:
        gammas, max_gamma, min_gamma = scm.get_gamma(a=a, x=x_test, key=key)
        ratio_u0, ratio_u1 = scm.get_density_ratios(a=a, x=x_test, key=key)
        prop_obs = scm.propensity_obs(a=a, x=x_test)
        # Plot
        plt.plot(x_test, gammas, label="Gamma " + key)
        #plt.plot(x_test, max_gamma, label="Max Gamma")
        #plt.plot(x_test, min_gamma, label="Min Gamma")
        #plt.plot(x_test, ratio_u0, label="Ratio u0" + key)
        #plt.plot(x_test, ratio_u1, label="Ratio u1" + key)
    plt.legend()
    plt.title("Oracle confounding")
    plt.show()