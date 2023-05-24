from data.data_structures import GSM_Dataset
from data.data_structures import Train_Dataset
from models.gmsm import GMSM
from models.neural import MLP, CondNormalizingFlow
import utils.utils as utils
import torch


# Generalized marginal sensitivity model for a single binary treatment
class GMSM_binary(GMSM):
    def __init__(self, causal_graph, y_type, models=None, scaling_params=None):
        super().__init__(causal_graph, y_type, models=models, scaling_params=scaling_params)

    def get_s(self, key, data: GSM_Dataset, gammas):
        # Calculate s- for a given key in M or Y
        if "propensity" not in self.models.keys():
            raise Exception("Propensity model not trained")
        a = data.data["a"].to(torch.int64)
        prop1 = self.predict_by_key("propensity", data)
        prop = torch.concat((1 - prop1, prop1), dim=1)
        prop_a = prop.gather(1, a)
        gammas = gammas.repeat(prop_a.size(0), 1)
        s_minus = 1 / ((1 - gammas) * prop_a + gammas)
        s_plus = 1 / ((1 - (1 / gammas)) * prop_a + (1 / gammas))
        #Outputs are of size n x len(gammas)
        return {"s_minus": s_minus, "s_plus": s_plus}


class GMSM_continuous(GMSM):
    def __init__(self, causal_graph, y_type, models=None, scaling_params=None):
        super().__init__(causal_graph, y_type, models=models, scaling_params=scaling_params)

    def get_s(self, key, data: GSM_Dataset, gammas):
        # Calculate s- for a given key in M or Y
        a = data.data["a"]
        gammas = gammas.repeat(a.size(0), 1)
        s_minus = 1 / gammas
        s_plus = gammas
        #Outputs are of size n x len(gammas)
        return {"s_minus": s_minus, "s_plus": s_plus}


class GMSM_weighted(GMSM):
    def __init__(self, causal_graph, y_type, weight_function, models=None, scaling_params=None):
        super().__init__(causal_graph, y_type, models=models, scaling_params=scaling_params)
        self.weight_function = weight_function

    def get_s(self, key, data: GSM_Dataset, gammas):
        weights = self.weight_function(data)
        # Calculate s- for a given key in M or Y
        gammas = gammas.repeat(weights.size(0), 1)
        s_minus = 1 / ((1 - gammas) * weights + gammas)
        s_plus = 1 / ((1 - (1 / gammas)) * weights + (1 / gammas))
        #Outputs are of size n x len(gammas)
        return {"s_minus": s_minus, "s_plus": s_plus}


