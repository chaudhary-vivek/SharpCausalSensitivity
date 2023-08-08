import torch
import pyreadr
import numpy as np
import utils.utils as utils
import random
from sklearn import preprocessing
from sklearn import model_selection
from data.data_structures import GSM_Dataset

# Code adjusted from: Jesson et al. 2022, "Scalable Sensitivity and Uncertainty Analyses for
# Causal-Effect Estimates of Continuous-Valued Interventions"

def load_data_ihdp(config):
    _CONTINUOUS_COVARIATES = [
        "bw",
        "b.head",
        "preterm",
        "birth.o",
        "nnhealth",
        "momage",
    ]

    _BINARY_COVARIATES = [
        "sex",
        "twin",
        "mom.lths",
        "mom.hs",
        "mom.scoll",
        "cig",
        "first",
        "booze",
        "drugs",
        "work.dur",
        "prenatal",
        "ark",
        "ein",
        "har",
        "mia",
        "pen",
        "tex",
        "was",
    ]

    _HIDDEN_COVARIATE = [
        "b.marr",
    ]

    # Load data
    data_path = utils.get_project_path() + "/data/data_ihdp/ihdp.RData"
    df = pyreadr.read_r(data_path)["ihdp"]

    # Make observational as per Hill 2011
    df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
    df = df[
        _CONTINUOUS_COVARIATES + _BINARY_COVARIATES + _HIDDEN_COVARIATE + ["treat"]
        ]
    # Standardize continuous covariates
    df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
        df[_CONTINUOUS_COVARIATES]
    )
    # Generate response surfaces
    seed = random.randint(0, 1000000)
    rng = np.random.default_rng(seed)
    x = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES]
    u = df[_HIDDEN_COVARIATE]
    t = df["treat"]
    beta_x = rng.choice(
        [0.0, 0.1, 0.2, 0.3, 0.4], size=(24,), p=[0.6, 0.1, 0.1, 0.1, 0.1]
    )
    beta_u = (
        rng.choice(
            [0.1, 0.2, 0.3, 0.4, 0.5], size=(1,), p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        if config["beta_u"] is None
        else np.asarray([config["beta_u"]])
    )
    mu0 = np.exp((x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u))
    #mu0 = 0
    df["mu0"] = mu0
    mu1 = (x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u)
    omega = (mu1[t == 1] - mu0[t == 1]).mean(0) - 4
    #omega = (mu1[t == 1]).mean(0)
    mu1 -= omega
    df["mu1"] = mu1
    eps = rng.normal(size=t.shape, scale=1, loc=0.0)
    y0 = mu0 + eps
    #y0 = eps
    df["y0"] = y0
    y1 = mu1 + eps
    #y1 = mu1 + np.exp(eps)
    df["y1"] = y1
    y = t * y1 + (1 - t) * y0
    df["y"] = y
    # Train test split
    df_train, df_test = model_selection.train_test_split(
        df, test_size=0.2, random_state=seed
    )

    # Set x, y, and t values
    hidden_confounding = config["hidden_confounding"]
    covars = _CONTINUOUS_COVARIATES + _BINARY_COVARIATES
    covars = covars + _HIDDEN_COVARIATE if not hidden_confounding else covars

    # Test data
    x_test = df_test[covars].to_numpy(dtype="float32")
    a_test = np.expand_dims(df_test["treat"].to_numpy(dtype="float32"), 1)
    mu0_test = np.expand_dims(df_test["mu0"].to_numpy(dtype="float32"), 1)
    mu1_test = np.expand_dims(df_test["mu1"].to_numpy(dtype="float32"), 1)
    y0_test = np.expand_dims(df_test["y0"].to_numpy(dtype="float32"), 1)
    y1_test = np.expand_dims(df_test["y1"].to_numpy(dtype="float32"), 1)
    y_test = np.expand_dims(df_test["y"].to_numpy(dtype="float32"), 1)

    d_test = GSM_Dataset(x=x_test, a=a_test, y=y_test, m=None, x_type="continuous", a_type="binary",
                         y_type="continuous")

    if config["validation"] == True:
        df_train, df_val = model_selection.train_test_split(
            df_train, test_size=0.3, random_state=seed
        )

        # Validation data
        x_val = df_val[covars].to_numpy(dtype="float32")
        a_val = np.expand_dims(df_val["treat"].to_numpy(dtype="float32"), 1)
        y_val = np.expand_dims(df_val["y"].to_numpy(dtype="float32"), 1)
        d_val = GSM_Dataset(x=x_val, a=a_val, y=y_val, m=None, x_type="continuous", a_type="binary",
                             y_type="continuous")

    else:
        d_val = None

    # Train data
    x_train = df_train[covars].to_numpy(dtype="float32")
    a_train = np.expand_dims(df_train["treat"].to_numpy(dtype="float32"), 1)
    y_train = np.expand_dims(df_train["y"].to_numpy(dtype="float32"), 1)
    d_train = GSM_Dataset(x=x_train, a=a_train, y=y_train, m=None, x_type="continuous", a_type="binary",
                            y_type="continuous")

    causal_graph = {"nodes": ["a", "y"], "edges": [("a", "y")]}

    return {"d_train": d_train, "d_val": d_val, "d_test": d_test, "causal_graph": causal_graph,
            "y0_test": y0_test, "y1_test": y1_test, "mu0_test": mu0_test, "mu1_test": mu1_test}