from data.simulated_scms import SCM_continuous, SCM_binary, SCM_continuous_weight
from data.data_structures import GSM_Dataset
from data.data_real import load_data_real

def get_datasets(config):
    if config["name"] == "sim_binary" or config["name"] == "sim_binary_m1" or config["name"] == "sim_binary_m2":
        scm = SCM_binary(config)
        # Trainig data
        [x, a, m1, m2, y] = scm.generate_data(n_samples=config["n_train"])
        # Validation data
        [x_val, a_val, m1_val, m2_val, y_val] = scm.generate_data(n_samples=config["n_val"])
        if config["name"] == "sim_binary":
            d_train = GSM_Dataset(x=x, a=a, m=None, y=y, x_type="continuous", a_type="binary",
                                  y_type="continuous")
            d_val = GSM_Dataset(x=x_val, a=a_val, m=None, y=y_val, x_type="continuous",
                                a_type="binary", y_type="continuous")
            causal_graph = {"nodes": ["a", "y"], "edges": [("a", "y")]}
        elif config["name"] == "sim_binary_m1":
            d_train = GSM_Dataset(x=x, a=a, m={"m1": m1}, y=y, x_type="continuous", a_type="binary",
                                  y_type="continuous")
            d_val = GSM_Dataset(x=x_val, a=a_val, m={"m1": m1_val}, y=y_val, x_type="continuous",
                                a_type="binary", y_type="continuous")
            causal_graph = {"nodes": ["a", "m1", "y"],
                            "edges": [("a", "m1"), ("a", "y"), ("m1", "y")]}
        elif config["name"] == "sim_binary_m2":
            d_train = GSM_Dataset(x=x, a=a, m={"m1": m1, "m2": m2}, y=y, x_type="continuous", a_type="binary",
                                  y_type="continuous")
            d_val = GSM_Dataset(x=x_val, a=a_val, m={"m1": m1_val, "m2": m2_val}, y=y_val, x_type="continuous",
                                a_type="binary", y_type="continuous")
            causal_graph = {"nodes": ["a", "m1", "m2", "y"],
                            "edges": [("a", "m1"), ("a", "m2"), ("a", "y"), ("m1", "m2"), ("m1", "y"), ("m2", "y")]}
        else:
            raise NotImplementedError
        return {"d_train": d_train, "d_val": d_val, "scm": scm, "causal_graph": causal_graph}

    elif config["name"] == "sim_continuous" or config["name"] == "sim_continuous_m1" or config["name"] == "sim_continuous_m2":
        scm = SCM_continuous(config)
        # Trainig data
        [x, a, m1, m2, y] = scm.generate_data(n_samples=config["n_train"])
        # Validation data
        [x_val, a_val, m1_val, m2_val, y_val] = scm.generate_data(n_samples=config["n_val"])
        if config["name"] == "sim_continuous":
            d_train = GSM_Dataset(x=x, a=a, m=None, y=y, x_type="continuous", a_type="continuous",
                                  y_type="continuous")
            d_val = GSM_Dataset(x=x_val, a=a_val, m=None, y=y_val, x_type="continuous",
                                a_type="continuous", y_type="continuous")
            causal_graph = {"nodes": ["a", "y"], "edges": [("a", "y")]}
        elif config["name"] == "sim_continuous_m1":
            d_train = GSM_Dataset(x=x, a=a, m={"m1": m1}, y=y, x_type="continuous", a_type="continuous",
                                  y_type="continuous")
            d_val = GSM_Dataset(x=x_val, a=a_val, m={"m1": m1_val}, y=y_val, x_type="continuous",
                                a_type="continuous", y_type="continuous")
            causal_graph = {"nodes": ["a", "m1", "y"],
                            "edges": [("a", "m1"), ("a", "y"), ("m1", "y")]}
        elif config["name"] == "sim_continuous_m2":
            d_train = GSM_Dataset(x=x, a=a, m={"m1": m1, "m2": m2}, y=y, x_type="continuous", a_type="continuous",
                                  y_type="continuous")
            d_val = GSM_Dataset(x=x_val, a=a_val, m={"m1": m1_val, "m2": m2_val}, y=y_val, x_type="continuous",
                                a_type="continuous", y_type="continuous")
            causal_graph = {"nodes": ["a", "m1", "m2", "y"],
                            "edges": [("a", "m1"), ("a", "m2"), ("a", "y"), ("m1", "m2"), ("m1", "y"), ("m2", "y")]}
        else:
            raise NotImplementedError

        return {"d_train": d_train, "d_val": d_val, "scm": scm, "causal_graph": causal_graph}

    elif config["name"] == "sim_continuous_weight":
        scm = SCM_continuous_weight(config)
        # Trainig data
        [x, a, _, _, y] = scm.generate_data(n_samples=config["n_train"])
        # Validation data
        [x_val, a_val, _, _, y_val] = scm.generate_data(n_samples=config["n_val"])
        d_train = GSM_Dataset(x=x, a=a, m=None, y=y, x_type="continuous", a_type="continuous",
                              y_type="continuous")
        d_val = GSM_Dataset(x=x_val, a=a_val, m=None, y=y_val, x_type="continuous",
                            a_type="continuous", y_type="continuous")
        causal_graph = {"nodes": ["a", "y"], "edges": [("a", "y")]}
        return {"d_train": d_train, "d_val": d_val, "scm": scm, "causal_graph": causal_graph}

    elif config["name"] == "real":
        return load_data_real(config)



