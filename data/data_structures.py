import torch
from torch.utils.data import Dataset
import numpy as np

#Dataset as input for a generalized sensitivity model
class GSM_Dataset(Dataset):
    # x is a numpy array of shape (n, d_x), a is a numpy array of shape (n, d_ai),
    # m is dictionary of numpy arrays of shape (n, d_mi), y is a numpy array of shape (n, 1)
    # if d_mi > 1, m is one-hot encoded, otherwise it is binary
    # causal_graph is a list of pairs (directed edges) (p,q), where p, q are dictionary keys of A, M, Y
    # x_type and a_type are dictionaries with keys corresponding to A, M, Y, and values "continuous" or "binary"
    # m entries are binary (one-hot encoded) and y is continuous/ binary
    
    def __init__(self, x, a, m, y, x_type=None, a_type=None, y_type=None):
        self.scaling_params = {"y": {"mean": 0, "sd": 1}}
        self.datatypes = {"x_type": x_type, "a_type": a_type, "y_type": y_type}
        #Convert to pytorch tensors
        self.data = {}
        if x is not None:
            self.data["x"] = torch.from_numpy(x.astype(np.float32))
        if a is not None:
            self.data["a"] = torch.from_numpy(a.astype(np.float32))
        if m is not None:
            self.data["m"] = {k: torch.from_numpy(v.astype(np.float32)) for k, v in m.items()}
        else:
            self.data["m"] = {}
        if y is not None:
            self.data["y"] = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.data["x"].size(0)
    
    def get_type_by_key(self, key):
        if key == "x":
            return self.datatypes["x_type"]
        if key == "a":
            return self.datatypes["a_type"]
        if key == "y":
            return self.datatypes["y_type"]
        if key in self.data["m"].keys():
            return "discrete"
        raise ValueError("Invalid key")

    def get_dims(self, keys):
        dims = {}
        for key in keys:
            if key == "x":
                dims["x"] = self.data["x"].size(1)
            if key == "y":
                dims["y"] = self.data["y"].size(1)
            if key == "a":
                dims[key] = self.data["a"].size(1)
            if key in self.data["m"].keys():
                dims[key] = self.data["m"][key].size(1)
        return dims

    #Input: list of integers, sets the values of the m variables to the corresponding values in m_list
    def set_m(self, m_list):
        if len(m_list) > len(self.data["m"].keys()):
            raise ValueError("Length of m_list must be equal or smaller to number of m variables")
        for i, key in enumerate(self.data["m"].keys()):
            if i < len(m_list):
                #One hot encoding
                if self.data["m"][key].size(1) > 1:
                    self.data["m"][key] = torch.zeros(self.data["m"][key].size())
                    self.data["m"][key][:, m_list[i]] = 1
                elif self.data["m"][key].size(1) == 1:
                    #Binary m
                    self.data["m"][key] = torch.full(size=self.data["m"][key].size(), fill_value=m_list[i])
                else:
                    raise ValueError("m must be a vector or matrix")

    #def __getitem__(self, index) -> dict:
    #    y = {"y" : self.data["y"][index]}
    #    x = {"x" : self.data["x"][index]}
    #    m = {k: v[index] for k, v in self.data["m"].items()}
    #    a = {k: v[index] for k, v in self.data["a"].items()}
    #    return {**x, **a, **m, **y}

    # Scaling------------------
    def scale_y(self):
        mean = torch.squeeze(torch.mean(self.data["y"], dim=0))
        sd = torch.squeeze(torch.std(self.data["y"], dim=0))
        self.data["y"] = self.__scale_vector(self.data["y"], mean, sd)
        self.scaling_params["y"]["mean"] = mean
        self.scaling_params["y"]["sd"] = sd

    def unscale_y(self, log_transform=False):
        mean = self.scaling_params["y"]["mean"]
        sd = self.scaling_params["y"]["sd"]
        self.data["y"] = self.__unscale_vector(self.data["y"], mean, sd)
        self.scaling_params["y"]["mean"] = 0
        self.scaling_params["y"]["sd"] = 1

    @staticmethod
    def __scale_vector(data, m, sd):
        return (data - m) / sd

    @staticmethod
    def __unscale_vector(data, m, sd):
        return (data * sd) + m

    def get_bootstrapped_dataset(self, n_samples):
        n = self.data["x"].size(0)
        idx = np.random.choice(n, n_samples, replace=True)
        return GSM_Dataset(x=self.data["x"][idx].detach().numpy(), a=self.data["a"][idx].detach().numpy(),
                           m={k: v[idx].detach().numpy() for k, v in self.data["m"].items()}, y=self.data["y"].detach().numpy(),
                           x_type=self.datatypes["x_type"], a_type=self.datatypes["a_type"], y_type=self.datatypes["y_type"])



#Dataset used to train pytorch saved_models
#Contains (concatinated) input and output tensors
class Train_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, index) -> dict:
        return {"x": self.x[index], "y": self.y[index]}

    def get_sizes(self):
        return {"n": self.y.size(0), "d_in": self.x.size(1), "d_out": self.y.size(1)}
