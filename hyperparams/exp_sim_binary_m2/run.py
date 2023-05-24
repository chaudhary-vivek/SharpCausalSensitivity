import utils.utils as utils
from hyperparams.tuning import run_hyper_tuning

if __name__ == "__main__":
    config_hyper = utils.load_yaml("/hyperparams/exp_sim_binary_m2/config")
    config_data = utils.load_yaml("/experiments/exp_sim_binary_m2/config")["data"]
    run_hyper_tuning(config_hyper, config_data)