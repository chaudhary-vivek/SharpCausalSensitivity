from data.data_structures import GSM_Dataset
from models.abstract import Sensitivity_model
from models.neural import MLP, CondNormalizingFlow
import torch
import numpy as np
from utils import utils
from data.data_structures import Train_Dataset


class CMSM(Sensitivity_model):
    def __init__(self, causal_graph, y_type, models=None, scaling_params=None):
        if models is None:
            self.models = {}
        else:
            self.models = models
        if scaling_params is None:
            self.scaling_params = {"mean": 0, "sd": 1}
        else:
            self.scaling_params = scaling_params
        self.causal_graph = causal_graph
        self.y_type = y_type

    def fit(self, config, d_train: GSM_Dataset, d_val=None, y_scale=False):
        # Scaling
        self.scaling_params = {"mean": 0, "sd": 1}
        if y_scale:
            d_train.scale_y()
            d_val.scale_y()
            self.scaling_params = d_train.scaling_params["y"]
        # Train/ validation data
        data_train = Train_Dataset(torch.concat([d_train.data["x"], d_train.data["a"]], dim=1), d_train.data["y"])
        if d_val is not None:
            data_val = Train_Dataset(torch.concat([d_val.data["x"], d_val.data["a"]], dim=1), d_val.data["y"])
        else:
            data_val = None
        # Fit outcome regression
        if "y_regression" not in self.models.keys():
            config_yreg = config["y_regression"]
            config_yreg["d_in"] = data_train.get_sizes()["d_in"]
            config_yreg["d_out"] = data_train.get_sizes()["d_out"]
            self.models["y_regression"] = utils.train_model(MLP(config=config_yreg, out_type=self.y_type), config_yreg,
                                                            data_train, data_val)["trained_model"]
        # Fit outcome distribution
        if "y" not in self.models.keys():
            config_y = config["y"]
            config_y["d_in"] = data_train.get_sizes()["d_in"]
            config_y["d_out"] = data_train.get_sizes()["d_out"]
            if self.y_type is "continuous":
                self.models["y"] = utils.train_model(CondNormalizingFlow(config=config_y), config_y,
                                                     data_train, data_val)["trained_model"]

    def get_mu(self, w, gamma_term, y_means, y_samples):
        I1 = torch.mean(w * (y_samples - y_means), dim=1, keepdim=True)
        I2 = torch.mean(w, dim=1, keepdim=True)
        return y_means + (I1 / (gamma_term + I2))

    def get_mu_plus(self, y_H, gamma_term, y_means, ysamples):
        diff = ysamples - y_H
        w = torch.heaviside(diff, torch.zeros_like(diff))
        return self.get_mu(w, gamma_term, y_means, ysamples)

    def get_mu_minus(self, y_H, gamma_term, y_means, ysamples):
        diff = y_H - ysamples
        w = torch.heaviside(diff, torch.zeros_like(diff))
        return self.get_mu(w, gamma_term, y_means, ysamples)

    def compute_bounds(self, x, a_int, gamma_dict, n_samples=500, n_search=1000, q=None):
        data = GSM_Dataset(x=x, a=np.full((x.shape[0], 1), a_int[0]), m=None, y=None)
        #Predict/ sample from saved_models
        y_samples = self.models["y"].sample(torch.concat([data.data["x"], data.data["a"]], dim=1), n_samples,
                                            self.scaling_params)
        y_means = self.models["y_regression"].predict(torch.concat([data.data["x"], data.data["a"]], dim=1),
                                                      self.scaling_params)
        #Compute bounds
        gamma_term = gamma_dict["y"].repeat(data.data["x"].size(0), 1)
        gamma_term = 1 / ((gamma_term * gamma_term) - 1)

        #Initial values
        mu_plus = torch.full_like(gamma_term, -1000)
        mu_minus = torch.full_like(gamma_term, 1000)
        y_plus = torch.zeros_like(gamma_term)
        y_minus = torch.zeros_like(gamma_term)

        #Construct search grid
        #y_samples_min = torch.min(y_samples, dim=1, keepdim=True)[0]
        #y_samples_max = torch.max(y_samples, dim=1, keepdim=True)[0]
        #search_grid = torch.zeros_like(y_samples)

        #Start grid search
        for y_H in torch.unbind(y_samples, dim=1):
            y_H = torch.unsqueeze(y_H, dim=1)
            kappa_plus = self.get_mu_plus(y_H, gamma_term, y_means, y_samples)
            kappa_minus = self.get_mu_minus(y_H, gamma_term, y_means, y_samples)
            idx_plus = kappa_plus > mu_plus
            mu_plus[idx_plus] = kappa_plus[idx_plus]
            y_plus[idx_plus] = y_H.repeat((1, gamma_term.size(1)))[idx_plus]
            idx_minus = kappa_minus < mu_minus
            mu_minus[idx_minus] = kappa_minus[idx_minus]
            y_minus[idx_minus] = y_H.repeat((1, gamma_term.size(1)))[idx_minus]

        Q_plus = torch.zeros_like(gamma_term)
        Q_minus = torch.zeros_like(gamma_term)
        for i in range(gamma_term.size(1)):
            Q_plus[:, i:i+1] = self.get_mu_plus(y_plus[:, i:i+1], gamma_term, y_means, y_samples)[:, i:i+1]
            Q_minus[:, i:i+1] = self.get_mu_minus(y_minus[:, i:i+1], gamma_term, y_means, y_samples)[:, i:i+1]
        return {"Q_plus": Q_plus, "Q_minus": Q_minus}

