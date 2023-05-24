from abc import ABC, abstractmethod
from data.data_structures import GSM_Dataset
import torch


class Sensitivity_model(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, config, d_train: GSM_Dataset, d_val=None):
        pass

    @abstractmethod
    def compute_bounds(self, x, a_int, gamma_dict, n_samples=500, q=None):
        pass

    # Estimate bounds on the average of an intervention
    def compute_bounds_average(self, x, a_int, gamma_dict, n_samples=500, q=None):
        Q = self.compute_bounds(x, a_int, gamma_dict, n_samples, q=q)
        return {"Q_plus": torch.mean(Q["Q_plus"], dim=0), "Q_minus": torch.mean(Q["Q_minus"], dim=0)}

    # Estimate bounds on the difference between two interventions
    def compute_bounds_difference(self, x, gamma_dict, n_samples=500, a_int1=1, a_int2=0, q=None):
        Q1 = self.compute_bounds(x, a_int1, gamma_dict, n_samples, q=q)
        Q2 = self.compute_bounds(x, a_int2, gamma_dict, n_samples, q=q)
        return {"Q_plus": Q1["Q_plus"] - Q2["Q_minus"], "Q_minus": Q1["Q_minus"] - Q2["Q_plus"]}

    # Estimate bounds on the average difference between two interventions
    def compute_bounds_diff_avg(self, x, gamma_dict, n_samples=500, a_int1=1, a_int2=0, q=None):
        Q = self.compute_bounds_difference(x, gamma_dict, n_samples, a_int1, a_int2, q=q)
        return {"Q_plus": torch.mean(Q["Q_plus"], dim=0, keepdim=True), "Q_minus": torch.mean(Q["Q_minus"], dim=0, keepdim=True)}
