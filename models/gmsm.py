from data.data_structures import GSM_Dataset
from data.data_structures import Train_Dataset
from models.neural import MLP, CondNormalizingFlow
from models.abstract import Sensitivity_model
import utils.utils as utils
import torch
import numpy as np


# Generalized marginal sensitivity model
# Causal graph is dictionary with keys "nodes" and "edges", is a causal graph on A, M, and Y
class GMSM(Sensitivity_model):
    def __init__(self, causal_graph, y_type, models=None, scaling_params=None):
        self.scaling_params = {"mean": 0, "sd": 1}
        self.causal_graph = causal_graph
        if models is None:
            self.models = {}
            self.m_dims = None
        else:
            self.models = models
            #create m_dims
            m_keys = [node for node in self.causal_graph["nodes"] if node not in ["a", "y"]]
            self.m_dims = {}
            for key in m_keys:
                if key in models.keys():
                    self.m_dims[key] = models[key].output_size
        if scaling_params is None:
            self.scaling_params = {"mean": 0, "sd": 1}
        else:
            self.scaling_params = scaling_params
        self.y_type = y_type

    def get_parent_keys(self, child_keys):
        # Get parent keys for a given child key
        parents = ["x"]
        for edge in self.causal_graph["edges"]:
            if edge[1] in child_keys:
                parents.append(edge[0])
        return parents

    def get_data_tensor(self, keys, dataset: GSM_Dataset):
        # Get concatinated tensor of data for a list of keys
        data_tensors = []
        if "x" in keys:
            data_tensors.append(dataset.data["x"])
        if "a" in keys and "a" in dataset.data.keys():
            data_tensors.append(dataset.data["a"])
        for key in keys:
            if key in dataset.data["m"].keys():
                data_tensors.append(dataset.data["m"][key])
        if "y" in keys and "y" in dataset.data.keys():
            data_tensors.append(dataset.data["y"])
        if len(data_tensors) > 0:
            return torch.concat(data_tensors, dim=1)
        else:
            return None

    def get_train_dataset(self, keys, data: GSM_Dataset):
        # Get dataset for training a model for a given key
        parents = self.get_parent_keys(keys)
        input_train = self.get_data_tensor(parents, data)
        output_train = self.get_data_tensor(keys, data)
        # Onehot encode output
        #if output_train is not None:
        #    if output_train.size(1) == 1 and keys[0] != "y":
        #        output_train = torch.cat((1 - output_train, output_train), dim=1)
        return Train_Dataset(input_train, output_train)

    def fit_distribution_by_keys(self, keys, config, d_train: GSM_Dataset, d_val=None, datatype="discrete"):
        # Train conditional distribution model for a single node in M, Y
        # Create dataset for training
        data_train = self.get_train_dataset(keys, d_train)
        if d_val is not None:
            data_val = self.get_train_dataset(keys, d_val)
        else:
            data_val = None
        # Create model: either MLP or Normalizing Flow
        config["d_in"] = data_train.get_sizes()["d_in"]
        config["d_out"] = data_train.get_sizes()["d_out"]
        if datatype == "continuous":
            model = CondNormalizingFlow(config=config)
        else:
            model = MLP(config=config, out_type="discrete")
        return utils.train_model(model, config, data_train, data_val)

    def fit_y_regression(self, config, d_train: GSM_Dataset, d_val=None):
        # Train outcome regression model for Y (used for gamma=0)
        # Create dataset for training
        data_train = self.get_train_dataset(["y"], d_train)
        if d_val is not None:
            data_val = self.get_train_dataset(["y"], d_val)
        else:
            data_val = None
        # Create model: MLP with continuous output type
        config["d_in"] = data_train.get_sizes()["d_in"]
        config["d_out"] = data_train.get_sizes()["d_out"]
        model = MLP(config=config, out_type="continuous")
        return utils.train_model(model, config, data_train, data_val)

    def fit(self, config, d_train: GSM_Dataset, d_val=None, y_scale=False):
        # Train saved_models for all keys in config, can include nodes in M, Y, and propensity model
        # Config saved_models is a dictionary of with keys in propoensity, M, Y and values being dictionaries of model configurations
        # Save dimensions of M
        m_keys = [node for node in self.causal_graph["nodes"] if node not in ["a", "y"]]
        self.m_dims = d_train.get_dims(m_keys)
        #Scaling
        self.scaling_params = {"mean": 0, "sd": 1}
        if y_scale:
            d_train.scale_y()
            d_val.scale_y()
            self.scaling_params = d_train.scaling_params["y"]
        for key in config.keys():
            if key not in self.models.keys():
                if key in list(d_train.data["m"].keys()):
                    train_results = self.fit_distribution_by_keys([key], config[key], d_train=d_train, d_val=d_val, datatype="discrete")
                elif key == "y":
                    train_results = self.fit_distribution_by_keys([key], config[key], d_train=d_train, d_val=d_val,
                                                                  datatype=d_train.datatypes["y_type"])
                elif key == "y_regression":
                    train_results = self.fit_y_regression(config[key], d_train=d_train, d_val=d_val)
                elif key == "propensity":
                    train_results = self.fit_distribution_by_keys("a", config[key], d_train=d_train, d_val=d_val,
                                                                  datatype=d_train.datatypes["a_type"])
                else:
                    raise ValueError("Invalid key in config: " + key)
                #Add trained model to GMSM
                self.models[key] = train_results["trained_model"]

    # ygrid is of shape (n_test, n_grid)
    def predict_by_key(self, key, d_test: GSM_Dataset, y_grid=None):
        if key == "propensity":
            data_test = self.get_train_dataset("a", d_test)
        elif key == "y_regression":
            data_test = self.get_train_dataset("y", d_test)
        else:
            data_test = self.get_train_dataset([key], d_test)

        if isinstance(self.models[key], MLP):
            return self.models[key].predict(data_test.x, self.scaling_params)
        elif isinstance(self.models[key], CondNormalizingFlow):
            return self.models[key].predict_density(data_test.x, y_grid, self.scaling_params)

    def sample_y(self, d_test: GSM_Dataset, n_samples=1):
        data_test = self.get_train_dataset(["y"], d_test)
        return self.models["y"].sample(data_test.x, n_samples, self.scaling_params)

    def get_s(self, key, data: GSM_Dataset, gammas):
        # Compute s_minus
        raise NotImplementedError

    def get_c(self, key, data: GSM_Dataset, gammas):
        s_dict = self.get_s(key=key, data=data, gammas=gammas)
        c_plus = ((1 - s_dict["s_minus"]) * s_dict["s_plus"]) / (s_dict["s_plus"] - s_dict["s_minus"])
        c_minus = ((1 - s_dict["s_plus"]) * s_dict["s_minus"]) / (s_dict["s_minus"] - s_dict["s_plus"])
        #fill nan values in c_plus and c_minus with 0 (this is the case when Gamma = 1, i.e., no unobserved confounding)
        c_plus[torch.isnan(c_plus)] = 0
        c_minus[torch.isnan(c_minus)] = 0
        return {"c_plus": c_plus, "c_minus": c_minus, "s_plus": s_dict["s_plus"], "s_minus": s_dict["s_minus"]}

    def m_combinations(self, m_sizes):
        n = len(m_sizes)
        combinations = [[0] * n]
        for i in range(n):
            if m_sizes[i] == 1:
                feasible_values = [0, 1]
            else:
                feasible_values = [j for j in range(m_sizes[i])]
            new_combinations = []
            for combination in combinations:
                for value in feasible_values:
                    new_combination = combination.copy()
                    new_combination[i] = value
                    new_combinations.append(new_combination)
            combinations = new_combinations
        return combinations

    def get_Qy_quantile_continuous(self, data: GSM_Dataset, c_y, n_samples=500, q=0.5):
        q_index_plus = torch.floor(c_y["c_plus"] * n_samples).long()
        q_index_minus = torch.floor(c_y["c_minus"] * n_samples).long()

        # Sample and sort
        y_samples = self.sample_y(d_test=data, n_samples=n_samples)
        y_samples = torch.sort(y_samples, dim=1)[0]

        # Quantiles for shifted distribution
        y_gamma_plus = y_samples.gather(dim=1, index=q_index_plus).unsqueeze(1).expand(-1, n_samples, -1)
        y_gamma_minus = y_samples.gather(dim=1, index=q_index_minus).unsqueeze(1).expand(-1, n_samples, -1)

        #indices where samples are below quantiles
        weight_plus = 1 / c_y["s_plus"]
        weight_minus = 1 / c_y["s_minus"]
        cdf_summands_plus = torch.where(y_samples.unsqueeze(-1) <= y_gamma_plus, weight_plus.unsqueeze(1).expand(-1, n_samples, -1),
                                        weight_minus.unsqueeze(1).expand(-1, n_samples, -1))
        cdf_plus = torch.cumsum(cdf_summands_plus, dim=1) / n_samples

        cdf_summands_minus = torch.where(y_samples.unsqueeze(-1) <= y_gamma_minus, weight_minus.unsqueeze(1).expand(-1, n_samples, -1),
                                        weight_plus.unsqueeze(1).expand(-1, n_samples, -1))
        cdf_minus = torch.cumsum(cdf_summands_minus, dim=1) / n_samples

        mask_q_plus = torch.where(cdf_plus <= q, 1, 0)
        idx_q_plus = torch.sum(mask_q_plus, dim=1) - 1
        masK_q_minus = torch.where(cdf_minus <= q, 1, 0)
        idx_q_minus = torch.sum(masK_q_minus, dim=1) - 1
        Qy_plus = y_samples.gather(dim=1, index=idx_q_plus)
        Qy_minus = y_samples.gather(dim=1, index=idx_q_minus)
        return {"Qy_plus": Qy_plus, "Qy_minus": Qy_minus}

    #Continuous outcome bounds for fixed m
    def get_Qy_mean_continuous(self, data: GSM_Dataset, c_y, n_samples=500):
        if "y_regression" not in self.models.keys():
            #Empirical quantile index
            q_index_plus = torch.floor(c_y["c_plus"] * n_samples).long()
            q_index_minus = torch.floor(c_y["c_minus"] * n_samples).long()

            # Sample and sort
            y_samples = self.sample_y(d_test=data, n_samples=n_samples)
            y_samples = torch.sort(y_samples, dim=1)[0]

            # Compute cumulative sum
            y_cum = torch.cumsum(y_samples, dim=1)
            # Select indices corresponding to the gammas (size is n_data x n_gammas)
            y_cum_gamma_plus = y_cum.gather(dim=1, index=q_index_plus)
            y_cum_gamma_minus = y_cum.gather(dim=1, index=q_index_minus)
            #Comulative sum over all samples
            y_cum_all = y_cum[:, -1].unsqueeze(1)

            Qy_plus = (y_cum_gamma_plus / (n_samples * c_y["s_plus"])) + \
                      ((y_cum_all - y_cum_gamma_plus) / (n_samples * c_y["s_minus"]))
            # if right most qunatile reached, set to bound of support (formula above breaks down)
            idx_boundary = q_index_plus == n_samples - 1
            max_support = torch.max(y_samples, dim=1)[0].unsqueeze(1).repeat((1, Qy_plus.size(1)))
            Qy_plus[idx_boundary] = max_support[idx_boundary]

            Qy_minus = (y_cum_gamma_minus / (n_samples * c_y["s_minus"])) + \
                      ((y_cum_all - y_cum_gamma_minus) / (n_samples * c_y["s_plus"]))
        #Predictions with regression network (for gamma = 1)
        else:
            y_pred = self.predict_by_key("y_regression", data)
            Qy_plus = y_pred
            Qy_minus = y_pred
        return {"Qy_plus": Qy_plus, "Qy_minus": Qy_minus}

    #Outcome bounds for all m, returns tensor of shape (n_data, n_gammas, dim_m1, dim_m2, ...)
    def get_Qy(self, data: GSM_Dataset, gammas, n_samples=500, q=None):
        # Get quantile information
        c_y = self.get_c(key="y", data=data, gammas=gammas)
        m_sizes = [self.m_dims[k] for k in self.m_dims.keys()]
        m_values = [2 if size == 1 else size for size in m_sizes]
        Qy_plus = torch.zeros([len(data), len(gammas)] + m_values)
        Qy_minus = torch.zeros([len(data), len(gammas)] + m_values)
        m_comb = self.m_combinations(m_sizes)
        for m in m_comb:
            data.set_m(m)
            if self.y_type == "continuous" and q is None:
                Qy_m = self.get_Qy_mean_continuous(data=data, c_y=c_y, n_samples=n_samples)
            elif self.y_type == "continuous" and q is not None:
                Qy_m = self.get_Qy_quantile_continuous(data=data, c_y=c_y, q=q, n_samples=n_samples)
            elif self.y_type == "discrete" or self.y_type == "binary":
                raise NotImplementedError
            else:
                raise ValueError("Invalid y_type: " + self.y_type)
            idx = (slice(None), slice(None)) + tuple(m)
            Qy_plus[idx] = Qy_m["Qy_plus"]
            Qy_minus[idx] = Qy_m["Qy_minus"]
        return {"Qy_plus": Qy_plus, "Qy_minus": Qy_minus}

    #Get the maximally shifted interventional distribution for discrete variables
    #permute_indices is a list of indices of length dim_key to permute target variable before weighting the predicted distribution
    def bound_distribution_discrete(self, key, data: GSM_Dataset, c, permutations=None):
        #Get model predictions
        prob_pred = self.predict_by_key(key=key, d_test=data)
        #One-hot encode for binary variables
        if prob_pred.size(1) == 1:
            prob_pred = torch.cat([1 - prob_pred, prob_pred], dim=1)

        #Expand prediction tensor to incorporate gamma dimension
        prob_pred = prob_pred.unsqueeze(1).expand(-1, c["c_plus"].size(1), -1)
        #Permute predictions if necessary
        if permutations is None:
            prob_pred_perm_plus = prob_pred
            prob_pred_perm_minus = prob_pred
        else:
            prob_pred_perm_plus = torch.gather(prob_pred, dim=2, index=permutations["plus"])
            prob_pred_perm_minus = torch.gather(prob_pred, dim=2, index=permutations["minus"])
        #Compute cumulative distrubution function (after permutation)
        F_plus = torch.cumsum(prob_pred_perm_plus, dim=2)
        F_minus = torch.cumsum(prob_pred_perm_minus, dim=2)
        F_plus_lagged = torch.cat([torch.zeros_like(F_plus[:, :, :1]), F_plus[:, :, :-1]], dim=2)
        F_minus_lagged = torch.cat([torch.zeros_like(F_minus[:, :, :1]), F_minus[:, :, :-1]], dim=2)

        #Apply inverse permutation to comulative distribution function
        if permutations is not None:
            # Compute inverse permutations
            perm_plus_inv = torch.sort(permutations["plus"], dim=2)[1]
            perm_minus_inv = torch.sort(permutations["minus"], dim=2)[1]
            F_plus = torch.gather(F_plus, dim=2, index=perm_plus_inv)
            F_minus = torch.gather(F_minus, dim=2, index=perm_minus_inv)
            F_plus_lagged = torch.gather(F_plus_lagged, dim=2, index=perm_plus_inv)
            F_minus_lagged = torch.gather(F_minus_lagged, dim=2, index=perm_minus_inv)

        #Weights
        weight_plus = 1 / c["s_plus"].unsqueeze(2).expand(-1, -1, prob_pred.size(2))
        weight_minus = 1 / c["s_minus"].unsqueeze(2).expand(-1, -1, prob_pred.size(2))
        # Quantiles
        c_plus = c["c_plus"].unsqueeze(2).expand(-1, -1, prob_pred.size(2))
        c_minus = c["c_minus"].unsqueeze(2).expand(-1, -1, prob_pred.size(2))
        #Compute reweighted distributions
        dist_plus = torch.where(F_plus < c_plus, prob_pred * weight_plus, torch.zeros_like(F_plus))
        dist_plus += torch.where(F_plus_lagged >= c_plus, prob_pred * weight_minus, torch.zeros_like(F_plus_lagged))
        dist_plus += torch.where((F_plus > c_plus) & (F_plus_lagged < c_plus),
                            (c_plus - F_plus_lagged) * weight_plus + (F_plus - c_plus) * weight_minus, torch.zeros_like(F_plus))

        dist_minus = torch.where(F_minus < c_minus, prob_pred * weight_minus, torch.zeros_like(F_minus))
        dist_minus += torch.where(F_minus_lagged >= c_minus, prob_pred * weight_plus, torch.zeros_like(F_minus_lagged))
        dist_minus += torch.where((F_minus > c_minus) & (F_minus_lagged < c_minus),
                            (c_minus - F_minus_lagged) * weight_minus + (F_minus - c_minus) * weight_plus, torch.zeros_like(F_minus))

        #Check whether reweighted distributions sums up to 1
        test_plus = torch.sum(dist_plus, dim=2)
        test_minus = torch.sum(dist_minus, dim=2)
        assert torch.allclose(test_plus, torch.ones_like(test_plus), rtol=0.001, atol=0.001)
        assert torch.allclose(test_minus, torch.ones_like(test_minus), rtol=0.001, atol=0.001)

        return {"plus": dist_plus, "minus": dist_minus}

    # x is a numpy array of shape (batch_size, dim_x)
    # a_int is a list of len num_m + 1, where the first element is a_1 and the last is a_l+1 (treatment interventions)
    # gamma_dict is a dictionary with keys "m_1", ..., "m_l", "y" and values numpy arrays containing the gammas for each MSM
    # n_samples is the number of samples to use for the Monte Carlo approximation (y)
    # q is the quantile to compute (default is None, which means that the mean is computed)
    def compute_bounds(self, x, a_int, gamma_dict, n_samples=500, q=None):
        # Check if model is fitted
        if self.m_dims is None:
            raise ValueError("m_dims is None, please fit the model first")
        #Set data: starting from last intervention a_int[-1] and an empty m
        m_keys = list(self.m_dims.keys())
        data = GSM_Dataset(x=x, a=np.full((x.shape[0], 1), a_int[-1]), m={k: np.zeros((x.shape[0], self.m_dims[k])) for k in m_keys}, y=None)
        # Compute bounds
        Qy = self.get_Qy(data=data, gammas=gamma_dict["y"], n_samples=n_samples, q=q)
        Q_plus = Qy["Qy_plus"]
        Q_minus = Qy["Qy_minus"]
        for i, key in reversed(list(enumerate(m_keys))):
            data.data["a"] = torch.full((x.shape[0], 1), a_int[i])
            # Get quantile information
            c_key = self.get_c(key=key, data=data, gammas=gamma_dict[key])
            # Create permutations
            permutation_plus = torch.sort(Q_plus, dim=-1)[1]
            permutation_minus = torch.sort(Q_minus, dim=-1)[1]
            # Create m distribution tensor of same shape as Q (last dimension is probability dimension)
            if i > 0:
                dist = {"plus": torch.zeros_like(Q_plus), "minus": torch.zeros_like(Q_minus)}
                m_sizes = [self.m_dims[k] for k in self.m_dims.keys()]
                m_comb_children = self.m_combinations(m_sizes[:i])
                for m_child in m_comb_children:
                    data.set_m(m_child)
                    idx = (slice(None), slice(None)) + tuple(m_child) + (slice(None),)
                    dist_m = self.bound_distribution_discrete(key=key, data=data, c=c_key,
                                            permutations={"plus": permutation_plus[idx], "minus": permutation_minus[idx]})
                    dist["plus"][idx] = dist_m["plus"]
                    dist["minus"][idx] = dist_m["minus"]
            else:
                dist = self.bound_distribution_discrete(key=key, data=data, c=c_key,
                                    permutations={"plus": permutation_plus, "minus": permutation_minus})
            Q_plus = torch.sum(Q_plus * dist["plus"], dim=-1)
            Q_minus = torch.sum(Q_minus * dist["minus"], dim=-1)
        return {"Q_plus": Q_plus, "Q_minus": Q_minus}

