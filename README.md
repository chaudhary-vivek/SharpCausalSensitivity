# SharpCausalSensitivity
Sharp bounds for generalized causal sensitivity analysis

#### Project structure 
- *data* contains the data generation files/ real-world data preprocessing
- *experiments* contains the code to run the experiments and the results
- *hyperparam* contains hyperparameters for the experiments and code for hyperparameter tuning
- *models* contains code for the sensitivity models, bound computation and estimation, and neural networks
- *notebooks* contains the code to create Figure 2 in the main paper


#### Requirements
The project is build with python 3.10 and uses the packages listed in the file `requirements.txt`. 

#### Reproducing the experiments
The scripts running the experiments are contained in the `/experiments` folder. There are two directories, one for each dataset (synthetic = `/exp_sim` and real-world = `/exp_real`). Most experiments can be configured by a `.yaml` configuration file. Here, parameters for data generation (e.g., sample size, covariate dimension) as well as the methods used may be adjusted. The following base methods are available (for details see Appendix E):

- `untrained`: untrained model,
- `oracle`: oracle unconstrained policy, 
- `oracle_af`: oracle action fair policy,
- `fpnet`: Our framework FairPol.

The `fpnet` methods has two sub-specifications:
- `action_fair`: is either `auf` (action unfair) or `af_conf` (action fairness using domain confusion)
- `value_fair`: is either `vuf` (value unfair), `vef` (envy-free), or `vmm` (max-min)


#### Reproducing hyperparameter tuning
The code for hyperparameter tuning is contained in the `/hyperparam` folder. The main script running the tuning is `main.py`. The subfolders contain the configuration files and optimal parameters for the different experiments (synthetic = `/expsim`, real-world data = `/exp_real`). The optimal parameters are stored as `.yaml` files in the `/nuisance` subfolder (for estimating nuisance parameters) or in the `/policy_nets` subfolder (for FairPol).
