import matplotlib.pyplot as plt
import utils.utils
import numpy as np
import seaborn as sns
import pandas as pd
import utils.utils as utils

def plot_bounds_over_x(x_test, gamma_dict, bounds_plus, bounds_minus, line_dict=None, save_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    #First plot lines
    if line_dict is not None:
        colors_lines = sns.color_palette("dark", len(line_dict.keys()))
        data_plot = pd.DataFrame({**{"x_test": x_test[:, 0]}, **line_dict})
        for i, key in enumerate(line_dict.keys()):
            sns.lineplot(data=data_plot, x="x_test", y=key, label=key, linewidth=2, color=colors_lines[i])

    # Then plot bounds
    num_gammas = len(gamma_dict[list(gamma_dict.keys())[0]])
    colors = sns.color_palette("flare", num_gammas)
    for i in range(num_gammas):
        keys = gamma_dict.keys()
        # Create label
        label = "Gamma "
        for key in keys:
            label += str(key) + "=" + str(gamma_dict[key][i].detach().numpy()) + " "
        if i == 0:
            plt.fill_between(x_test[:, 0], bounds_plus[:, 0], bounds_minus[:, 0], alpha=0.5, label=label, color=colors[i])
        else:
            # Fill between two times: bounds_plus[:, i-1] and bounds_plus[:, i], and bounds_minus[:, i-1] and bounds_minus[:, i]
            # Use the same label and color for both
            plt.fill_between(x_test[:, 0], bounds_plus[:, i - 1], bounds_plus[:, i], alpha=0.5, label=label, color=colors[i])
            plt.fill_between(x_test[:, 0], bounds_minus[:, i - 1], bounds_minus[:, i], alpha=0.5, color=colors[i])
    plt.legend()
    plt.show()


def plot_bounds_scm(gsm, scm, gamma_dict, a_int, x_test=None, n_samples=1000, n_samples_oracle=1000,
                       q=None, bootstrap=False, path_rel=None):
    if x_test is None:
        x_test = np.expand_dims(np.linspace(-1, 1, 200), axis=1)
    #Compute bounds
    if bootstrap:
        bounds = gsm.compute_bounds_ensemble(x_test, gamma_dict, a_int1=a_int, n_samples=n_samples, alpha=0.05, q=q,
                                             average=False)
        bounds_plus = bounds["Q_plus"]["mean"].detach().numpy()
        bounds_minus = bounds["Q_minus"]["mean"].detach().numpy()
    else:
        bounds = gsm.compute_bounds(x_test, a_int, gamma_dict, n_samples=n_samples, q=q)
        bounds_plus = bounds["Q_plus"].detach().numpy()
        bounds_minus = bounds["Q_minus"].detach().numpy()

    # Oracle effect
    if "m2" in gsm.causal_graph["nodes"]:
        y_int = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1], a3_int=a_int[2])
    elif "m1" in gsm.causal_graph["nodes"]:
        y_int = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1])
    else:
        y_int = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0])

    # Plot
    data_plot = pd.DataFrame({"x_test": x_test[:, 0], "y_int": y_int[:, 0], "gamma0": bounds_plus[:, 0]})
    # Create a Seaborn plot
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=data_plot, x="x_test", y="y_int", label="Oracle", color="darkred", linewidth=3)

    num_gammas = len(gamma_dict[list(gamma_dict.keys())[0]])
    colors = sns.color_palette("crest", num_gammas)
    legend_labels = {"y": "$\Gamma_Y$", "m1": "$\Gamma_{M_1}$", "m2": "$\Gamma_{M_2}$"}
    for i in range(num_gammas):
        keys = gamma_dict.keys()
        # Create label
        label = ""
        for key in keys:
            label += legend_labels[key] + "=" + str(gamma_dict[key][i].detach().numpy()) + " "
        if i == 0:
            sns.lineplot(data=data_plot, x="x_test", y="gamma0", label=label, color="darkblue", linewidth=2)
        else:
            # Fill between two times: bounds_plus[:, i-1] and bounds_plus[:, i], and bounds_minus[:, i-1] and bounds_minus[:, i]
            # Use the same label and color for both
            plt.fill_between(x_test[:, 0], bounds_plus[:, i - 1], bounds_plus[:, i], alpha=0.5, label=label,
                             color=colors[i])
            plt.fill_between(x_test[:, 0], bounds_minus[:, i - 1], bounds_minus[:, i], alpha=0.5, color=colors[i])
    plt.xlabel("x")
    plt.ylabel("Effect")
    plt.legend()
    if path_rel is not None:
        plt.savefig(utils.get_project_path() + path_rel)
    plt.show()


def plot_bounds_scm_quantile(gsm, scm, gamma_dict, a_int, x_test=None, n_samples=1000, n_samples_oracle=1000,
                       q=None, bootstrap=False, path_rel=None, legend_loc=None):
    if x_test is None:
        x_test = np.expand_dims(np.linspace(-1, 1, 200), axis=1)
    #Compute bounds
    if bootstrap:
        bounds = gsm.compute_bounds_ensemble(x_test, gamma_dict, a_int1=a_int, n_samples=n_samples, alpha=0.05, q=q,
                                             average=False)
        bounds_plus = bounds["Q_plus"]["mean"].detach().numpy()
        bounds_minus = bounds["Q_minus"]["mean"].detach().numpy()
    else:
        bounds = gsm.compute_bounds(x_test, a_int, gamma_dict, n_samples=n_samples, q=q)
        bounds_plus = bounds["Q_plus"].detach().numpy()
        bounds_minus = bounds["Q_minus"].detach().numpy()

    # Oracle effect
    if "m2" in gsm.causal_graph["nodes"]:
        y_int03 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1], a3_int=a_int[2], q=0.3)
        y_int05 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1], a3_int=a_int[2], q=0.5)
        y_int07 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1], a3_int=a_int[2], q=0.7)
    elif "m1" in gsm.causal_graph["nodes"]:
        y_int03 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1], q=0.3)
        y_int05 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1], q=0.5)
        y_int07 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], a2_int=a_int[1], q=0.7)
    else:
        y_int03 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], q=0.3)
        y_int05 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], q=0.5)
        y_int07 = scm.get_true_effect(x_test, n_samples=n_samples_oracle, a1_int=a_int[0], q=0.7)

    # Plot
    data_plot = pd.DataFrame({"x_test": x_test[:, 0], "y_int03": y_int03[:, 0], "y_int05": y_int05[:, 0], "y_int07": y_int07[:, 0], "gamma0": bounds_plus[:, 0]})
    # Create a Seaborn plot
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=data_plot, x="x_test", y="y_int03", label=r"Oracle $\alpha = 0.3$", color="brown", linewidth=3)
    sns.lineplot(data=data_plot, x="x_test", y="y_int05", label=r"Oracle $\alpha = 0.5$", color="darkred", linewidth=3)
    sns.lineplot(data=data_plot, x="x_test", y="y_int07", label=r"Oracle $\alpha = 0.7$", color="purple", linewidth=3)

    num_gammas = len(gamma_dict[list(gamma_dict.keys())[0]])
    colors = sns.color_palette("crest", num_gammas)
    legend_labels = {"y": "$\Gamma_Y$", "m1": "$\Gamma_{M_1}$", "m2": "$\Gamma_{M_2}$"}
    for i in range(num_gammas):
        keys = gamma_dict.keys()
        # Create label
        label = ""
        for key in keys:
            label += legend_labels[key] + "=" + str(gamma_dict[key][i].detach().numpy()) + " "
        if i == 0:
            sns.lineplot(data=data_plot, x="x_test", y="gamma0", label=label, color="darkblue", linewidth=2)
        else:
            # Fill between two times: bounds_plus[:, i-1] and bounds_plus[:, i], and bounds_minus[:, i-1] and bounds_minus[:, i]
            # Use the same label and color for both
            plt.fill_between(x_test[:, 0], bounds_plus[:, i - 1], bounds_plus[:, i], alpha=0.5, label=label,
                             color=colors[i])
            plt.fill_between(x_test[:, 0], bounds_minus[:, i - 1], bounds_minus[:, i], alpha=0.5, color=colors[i])
    plt.xlabel("x")
    plt.ylabel("Effect")
    if legend_loc is None:
        plt.legend()
    else:
        plt.legend(loc=legend_loc)
    if path_rel is not None:
        plt.savefig(utils.get_project_path() + path_rel)
    plt.show()

def plot_gamma_scm(scm, a_list, x_test=None, path_rel=None, binary=False):
    if x_test is None:
        x_test = np.expand_dims(np.linspace(-1, 1, 200), axis=1)

    colors = ["darkred", "darkgreen", "darkblue"]

    #Plotting
    sns.set(font_scale=1.8, style="whitegrid")
    plt.figure(figsize=(8, 6))
    if not binary:
        if len(a_list) == 1:
            labels = [r"$\Gamma^\ast_Y$"]
        elif len(a_list) == 2:
            labels = [r"$\Gamma^\ast_{M_1}$", r"$\Gamma^\ast_Y$"]
        elif len(a_list) == 3:
            labels = [r"$\Gamma^\ast_{M_1}$", r"$\Gamma^\ast_{M_2}$", r"$\Gamma^\ast_Y$"]
        else:
            raise ValueError("Too many treatments")
        for i, a in enumerate(a_list):
            gammas, max_gamma, min_gamma = scm.get_gamma(a=a, x=x_test, key="y")
            sns.lineplot(x=x_test[:, 0], y=gammas[:, 0], label=labels[i], linewidth=3, color=colors[i])
    else:
        if len(a_list) == 1:
            label = r"$\Gamma^\ast_Y$"
        elif len(a_list) == 2:
            label = r"$\Gamma^\ast_{M_1} = \Gamma^\ast_Y$"
        elif len(a_list) == 3:
            label = r"$\Gamma^\ast_{M_1} = \Gamma^\ast_{M_2} = \Gamma^\ast_Y$"
        else:
            raise ValueError("Too many treatments")
        prop_obs = scm.propensity_obs(a=1, x=x_test)
        sns.lineplot(x=x_test[:, 0], y=prop_obs[:, 0], label="Observed propensity score", linewidth=3, color="darkred")
        gammas, max_gamma, min_gamma = scm.get_gamma(a=1, x=x_test, key="y")
        sns.lineplot(x=x_test[:, 0], y=gammas[:, 0], label=label, linewidth=3, color="darkblue")
    plt.legend()
    if path_rel is not None:
        plt.savefig(utils.get_project_path() + path_rel)
    plt.show()


def plot_bounds_real(gsm, gamma_dict, a_int1, a_int2, x_test, n_samples=1000, q=None, bootstrap=False, path_rel=None):

    #Compute bounds
    if bootstrap:
        bounds = gsm.compute_bounds_ensemble(x_test, gamma_dict, a_int1=a_int1, a_int2=a_int2, n_samples=n_samples, alpha=0.8, q=q,
                                             average=True)
        bounds_plus_mean = bounds["Q_plus"]["mean"].detach().numpy().squeeze()
        bounds_minus_mean = bounds["Q_minus"]["mean"].detach().numpy().squeeze()
        bounds_plus_std = bounds["Q_plus"]["std"].detach().numpy().squeeze()
        bounds_minus_std = bounds["Q_minus"]["std"].detach().numpy().squeeze()
    else:
        raise NotImplementedError

    # Create a Seaborn plot
    gammas = gamma_dict["m1"].detach().numpy()
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=gammas, y=bounds_plus_mean, color="darkred", linewidth=2)
    sns.lineplot(x=gammas, y=bounds_minus_mean, color="darkred", linewidth=2)

    plt.fill_between(gammas, bounds_plus_mean + bounds_plus_std, bounds_plus_mean - bounds_plus_std, alpha=0.3, color="darkred")
    plt.fill_between(gammas, bounds_minus_mean + bounds_minus_std, bounds_minus_mean - bounds_minus_std, alpha=0.3, color="darkred")
    plt.xlabel("$\Gamma_{M}$")
    plt.ylabel("Effect size")
    plt.legend()
    if path_rel is not None:
        plt.savefig(utils.get_project_path() + path_rel)
    plt.show()
