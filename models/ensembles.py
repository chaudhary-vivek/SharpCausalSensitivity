import torch

class SensitivityEnsemble():
    def __init__(self, causal_graph, y_type, model_class, model_list=None, scaling_params=None):
        self.model_class = model_class
        if model_list is None:
            self.ensemble = []
        else:
            self.ensemble = [model_class(causal_graph, y_type, models) for models in model_list]
        if scaling_params is None:
            self.scaling_params = {"mean": 0, "sd": 1}
        else:
            self.scaling_params = scaling_params
            for model in self.ensemble:
                model.scaling_params = scaling_params
        self.causal_graph = causal_graph
        self.y_type = y_type

    def fit_ensemble(self, config, d_train, d_val=None, y_scale=False, k=5, resample=True):
        # Scaling
        self.scaling_params = {"mean": 0, "sd": 1}
        if y_scale:
            d_train.scale_y()
            if d_val is not None:
                d_val.scale_y()
            self.scaling_params = d_train.scaling_params["y"]
        for i in range(k):
            if resample:
                data_bootstrapped = d_train.get_bootstrapped_dataset(d_train.data["x"].size(0))
            else:
                data_bootstrapped = d_train
            model = self.model_class(self.causal_graph, self.y_type)
            model.fit(config, data_bootstrapped, d_val, y_scale=False)
            if y_scale:
                model.scaling_params = self.scaling_params
            self.ensemble.append(model)

    def compute_bounds_ensemble(self, x, gamma_dict, a_int1=1, a_int2=None, n_samples=500, alpha=0.05, q=None, average=False):
        bounds = []
        for model in self.ensemble:
            if a_int2 is None and average is False:
                bounds.append(model.compute_bounds(x, a_int1, gamma_dict, n_samples, q=q))
            elif a_int2 is None and average is True:
                bounds.append(model.compute_bounds_average(x, a_int1, gamma_dict, n_samples, q=q))
            elif a_int2 is not None and average is False:
                bounds.append(model.compute_bounds_difference(x, gamma_dict, n_samples, a_int1, a_int2, q=q))
            elif a_int2 is not None and average is True:
                bounds.append(model.compute_bounds_diff_avg(x, gamma_dict, n_samples, a_int1, a_int2, q=q))
            else:
                raise ValueError("Invalid arguments")
        Q_plus_b = torch.concat([torch.unsqueeze(Q["Q_plus"], 2) for Q in bounds], dim=2)
        Q_minus_b = torch.concat([torch.unsqueeze(Q["Q_minus"], 2) for Q in bounds], dim=2)
        #Compute percentile bootstrap bounds
        Q_plus_upper = torch.quantile(Q_plus_b, 1 - alpha / 2, 2)
        Q_plus_median = torch.quantile(Q_plus_b, 0.5, 2)
        Q_plus_lower = torch.quantile(Q_plus_b, alpha / 2, 2)

        Q_minus_upper = torch.quantile(Q_minus_b, 1 - alpha / 2, 2)
        Q_minus_median = torch.quantile(Q_minus_b, 0.5, 2)
        Q_minus_lower = torch.quantile(Q_minus_b, alpha / 2, 2)

        Q_plus_mean = torch.mean(Q_plus_b, 2)
        Q_plus_std = torch.std(Q_plus_b, 2)
        Q_minus_mean = torch.mean(Q_minus_b, 2)
        Q_minus_std = torch.std(Q_minus_b, 2)

        Q_plus_dict = {"mean": Q_plus_mean, "upper": Q_plus_upper, "lower": Q_plus_lower, "median": Q_plus_median, "std": Q_plus_std}
        Q_minus_dict = {"mean": Q_minus_mean, "upper": Q_minus_upper, "lower": Q_minus_lower, "median": Q_minus_median, "std": Q_minus_std}

        return {"Q_plus": Q_plus_dict, "Q_minus": Q_minus_dict}