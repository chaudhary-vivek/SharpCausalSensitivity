from abc import ABC, abstractmethod
from scipy.stats import beta
import numpy
import numpy as np


class DataSCM(ABC):
    def __init__(self, config):
        self.config = config

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @abstractmethod
    def f_a(self, x, u_m1, u_m2, u_y):
        pass

    @abstractmethod
    def f_m1(self, x, a, u_m1, eps_m1):
        pass

    @abstractmethod
    def f_m2(self, x, a, m1, u_m2, eps_m2):
        pass

    @abstractmethod
    def f_y(self, x, a, m1, m2, u_y, eps_y):
        pass

    @abstractmethod
    def propensity_full(self, a, x, u_m1, u_m2, u_y):
        pass

    def propensity_part(self, a, x, u, key="m1"):
        u0 = np.zeros((x.shape[0], 1))
        u1 = np.ones((x.shape[0], 1))
        if key == "m1":
            p00 = self.propensity_full(a, x, u, u0, u0)
            p01 = self.propensity_full(a, x, u, u0, u1)
            p10 = self.propensity_full(a, x, u, u1, u0)
            p11 = self.propensity_full(a, x, u, u1, u1)
        elif key == "m2":
            p00 = self.propensity_full(a, x, u0, u, u0)
            p01 = self.propensity_full(a, x, u0, u, u1)
            p10 = self.propensity_full(a, x, u1, u, u0)
            p11 = self.propensity_full(a, x, u1, u, u1)
        elif key == "y":
            p00 = self.propensity_full(a, x, u0, u0, u)
            p01 = self.propensity_full(a, x, u0, u1, u)
            p10 = self.propensity_full(a, x, u1, u0, u)
            p11 = self.propensity_full(a, x, u1, u1, u)
        else:
            raise ValueError("Key must be m1, m2 or y")
        return (p00 + p01 + p10 + p11) / 4

    def propensity_obs(self, a, x):
        u0 = np.zeros((x.shape[0], 1))
        u1 = np.ones((x.shape[0], 1))
        p000 = self.propensity_full(a, x, u0, u0, u0)
        p001 = self.propensity_full(a, x, u0, u0, u1)
        p010 = self.propensity_full(a, x, u0, u1, u0)
        p100 = self.propensity_full(a, x, u1, u0, u0)
        p011 = self.propensity_full(a, x, u0, u1, u1)
        p101 = self.propensity_full(a, x, u1, u0, u1)
        p110 = self.propensity_full(a, x, u1, u1, u0)
        p111 = self.propensity_full(a, x, u1, u1, u1)
        return (p000 + p001 + p010 + p100 + p011 + p101 + p110 + p111) / 8

    # The following methods express gamma in terms of density ratio when the MSM bounds are attained (max or min)
    @abstractmethod
    def get_gamma_maxratio(self, maxratio, prop_obs):
        pass

    @abstractmethod
    def get_gamma_minratio(self, minratio, prop_obs):
        pass

    def get_density_ratios(self, a, x, key="m1"):
        # Create propensity ratios both for u=0 and u=1
        prop_u0 = self.propensity_part(a, x, u=np.zeros((x.shape[0], 1)), key=key)
        prop_u1 = self.propensity_part(a, x, u=np.ones((x.shape[0], 1)), key=key)
        prop_obs = self.propensity_obs(a, x)
        # Ratios are vectors of same length as x/ a
        ratio_u0 = prop_u0 / prop_obs
        ratio_u1 = prop_u1 / prop_obs
        return ratio_u0, ratio_u1

    def get_gamma(self, a, x, key="m1"):
        # Create propensity ratios both for u=0 and u=1
        prop_obs = self.propensity_obs(a, x)
        # Ratios are vectors of same length as x/ a
        ratio_u0, ratio_u1 = self.get_density_ratios(a, x, key=key)
        # Maximum/ minimum ratios
        max_ratio = np.maximum(ratio_u0, ratio_u1)
        min_ratio = np.minimum(ratio_u0, ratio_u1)
        # Worst case gammas for maximum/ minimum ratios
        max_gamma = self.get_gamma_maxratio(max_ratio, prop_obs)
        min_gamma = self.get_gamma_minratio(min_ratio, prop_obs)
        # Worst case gammas
        gamma = np.maximum(max_gamma, min_gamma)
        return gamma, max_gamma, min_gamma

    def sample_noise(self, n_samples):
        # Sample latent variables
        X = np.random.uniform(-1, 1, (n_samples, self.config["d_x"]))
        U_m1 = np.random.binomial(1, 0.5, (n_samples, 1))
        U_m2 = np.random.binomial(1, 0.5, (n_samples, 1))
        U_y = np.random.binomial(1, 0.5, (n_samples, 1))
        eps_m1 = np.random.normal(0, 1, (n_samples, 1))
        eps_m2 = np.random.normal(0, 1, (n_samples, 1))
        eps_y = np.random.normal(0, 1, (n_samples, 1))
        return [X, U_m1, U_m2, U_y, eps_m1, eps_m2, eps_y]

    # Generates data from the SCM (with potential interventions for path specific effects)
    def generate_data(self, n_samples, int_x=None, a1_int=None, a2_int=None, a3_int=None):
        x, u_m1, u_m2, u_y, eps_m1, eps_m2, eps_y = self.sample_noise(n_samples)
        if int_x is not None:
            x = int_x
        a = self.f_a(x, u_m1, u_m2, u_y)
        if a1_int is not None:
            a = numpy.full_like(a, a1_int)
        m1 = self.f_m1(x, a, u_m1, eps_m1)
        if a2_int is not None:
            a = numpy.full_like(a, a2_int)
        m2 = self.f_m2(x, a, m1, u_m2, eps_m2)
        if a3_int is not None:
            a = numpy.full_like(a, a3_int)
        y = self.f_y(x, a, m1, m2, u_y, eps_y)

        return [x, a, m1, m2, y]

    # Computes the ground truth causal effect/ conditional mean for given x, by sampling the interventional/
    # observational distribution and averaging
    def get_true_effect(self, x, n_samples, a1_int=None, a2_int=None, a3_int=None, q=None):
        # create x of size (x.shape[0] * n_samples, 1) by repeating each entry n_sample times
        x = np.repeat(x, n_samples, axis=0)
        [_, _, _, _, int_y] = self.generate_data(n_samples=x.shape[0], int_x=x, a1_int=a1_int, a2_int=a2_int, a3_int=a3_int)
        int_y = np.reshape(int_y, (int(x.shape[0] / n_samples), n_samples))
        if q is None:
            return np.mean(int_y, axis=1, keepdims=True)
        else:
            return np.quantile(int_y, q, axis=1, keepdims=True)

    #Compute the conditional excpectations in population (replacing all do opratiors with conditioning)
    def get_conditional_effect(self, x, n_samples, a1_cond=None, a2_cond=None, a3_cond=None, a_type="binary", tol_a=0.05):
        # create x of size (x.shape[0] * n_samples, 1) by repeating each entry n_sample times
        x = np.repeat(x, n_samples, axis=0)
        [_, a, m1, m2, y] = self.generate_data(n_samples=x.shape[0], int_x=x)
        a = np.reshape(a, (int(x.shape[0] / n_samples), n_samples))
        m1 = np.reshape(m1, (int(x.shape[0] / n_samples), n_samples))
        m2 = np.reshape(m2, (int(x.shape[0] / n_samples), n_samples))
        y = np.reshape(y, (int(x.shape[0] / n_samples), n_samples))
        # Return mean y if no conditioning
        if a1_cond is None and a2_cond is None and a3_cond is None:
            return np.mean(y, axis=1, keepdims=True)
        # Conditioning only once ( = ITE)
        if a1_cond is not None and a2_cond is None and a3_cond is None:
            if a_type == "binary":
                ind_a1 = np.where(a == a1_cond, 1, 0)
            else:
                ind_a1 = np.where(np.abs(a - a1_cond) < tol_a, 1, 0)
            return np.sum(y * ind_a1, axis=1, keepdims=True) / np.sum(ind_a1, axis=1, keepdims=True)
        if a1_cond is not None and a2_cond is not None and a3_cond is None:
            if a_type == "binary":
                ind_a1 = np.where(a == a1_cond, 1, 0)
                ind_a2 = np.where(a == a2_cond, 1, 0)
            else:
                ind_a1 = np.where(np.abs(a - a1_cond) < tol_a, 1, 0)
                ind_a2 = np.where(np.abs(a - a2_cond) < tol_a, 1, 0)
            ind_m1eq0 = np.where(m1 == 0, 1, 0)
            y_a2_m1eq0 = np.sum(y * ind_a2 * ind_m1eq0, axis=1, keepdims=True) / np.sum(ind_a2 * ind_m1eq0, axis=1, keepdims=True)
            y_a2_m1eq1 = np.sum(y * ind_a2 * (1 - ind_m1eq0), axis=1, keepdims=True) / np.sum(ind_a2 * (1 - ind_m1eq0), axis=1, keepdims=True)
            m1_a0 = np.sum(m1 * ind_a1, axis=1, keepdims=True) / np.sum(ind_a1, axis=1, keepdims=True)
            return y_a2_m1eq0 * (1 - m1_a0) + y_a2_m1eq1 * m1_a0
        if a1_cond is not None and a2_cond is not None and a3_cond is not None:
            if a_type == "binary":
                ind_a1 = np.where(a == a1_cond, 1, 0)
                ind_a2 = np.where(a == a2_cond, 1, 0)
                ind_a3 = np.where(a == a3_cond, 1, 0)
            else:
                ind_a1 = np.where(np.abs(a - a1_cond) < tol_a, 1, 0)
                ind_a2 = np.where(np.abs(a - a2_cond) < tol_a, 1, 0)
                ind_a3 = np.where(np.abs(a - a3_cond) < tol_a, 1, 0)
            ind_m1eq0 = np.where(m1 == 0, 1, 0)
            ind_m2eq0 = np.where(m2 == 0, 1, 0)
            y_a3_m2eq0_m1eq0 = np.sum(y * ind_a3 * ind_m2eq0 * ind_m1eq0, axis=1, keepdims=True) / np.sum(ind_a3 * ind_m2eq0 * ind_m1eq0, axis=1, keepdims=True)
            y_a3_m2eq0_m1eq1 = np.sum(y * ind_a3 * ind_m2eq0 * (1 - ind_m1eq0), axis=1, keepdims=True) / np.sum(ind_a3 * ind_m2eq0 * (1 - ind_m1eq0), axis=1, keepdims=True)
            y_a3_m2eq1_m1eq0 = np.sum(y * ind_a3 * (1 - ind_m2eq0) * ind_m1eq0, axis=1, keepdims=True) / np.sum(ind_a3 * (1 - ind_m2eq0) * ind_m1eq0, axis=1, keepdims=True)
            y_a3_m2eq1_m1eq1 = np.sum(y * ind_a3 * (1 - ind_m2eq0) * (1 - ind_m1eq0), axis=1, keepdims=True) / np.sum(ind_a3 * (1 - ind_m2eq0) * (1 - ind_m1eq0), axis=1, keepdims=True)
            m2_a2_m1eq0 = np.sum(m2 * ind_a2 * ind_m1eq0, axis=1, keepdims=True) / np.sum(ind_a2 * ind_m1eq0, axis=1, keepdims=True)
            m2_a2_m1eq1 = np.sum(m2 * ind_a2 * (1 - ind_m1eq0), axis=1, keepdims=True) / np.sum(ind_a2 * (1 - ind_m1eq0), axis=1, keepdims=True)
            m1_a0 = np.sum(m1 * ind_a1, axis=1, keepdims=True) / np.sum(ind_a1, axis=1, keepdims=True)
            return y_a3_m2eq0_m1eq0 * (1 - m2_a2_m1eq0) * (1 - m1_a0) + y_a3_m2eq0_m1eq1 * (1 - m2_a2_m1eq1) * m1_a0 + \
                      y_a3_m2eq1_m1eq0 * m2_a2_m1eq0 * (1 - m1_a0) + y_a3_m2eq1_m1eq1 * m2_a2_m1eq1 * m1_a0

    #get conditional distribution of m given history
    def get_conditional_dist(self, x, n_samples, a_cond=1, m1_cond=1, key="m2"):
        x = np.repeat(x, n_samples, axis=0)
        [_, a, m1, m2, _] = self.generate_data(n_samples=x.shape[0], int_x=x)
        a = np.reshape(a, (int(x.shape[0] / n_samples), n_samples))
        m1 = np.reshape(m1, (int(x.shape[0] / n_samples), n_samples))
        m2 = np.reshape(m2, (int(x.shape[0] / n_samples), n_samples))
        ind_a = np.where(a == a_cond, 1, 0)
        if key == "m2":
            ind_m1 = np.where(m1 == m1_cond, 1, 0)
            return np.sum(m2 * ind_a * ind_m1, axis=1, keepdims=True) / np.sum(ind_a * ind_m1, axis=1,
                                                                                      keepdims=True)
        elif key == "m1":
            return np.sum(m1 * ind_a, axis=1, keepdims=True) / np.sum(ind_a, axis=1, keepdims=True)
        else:
            raise ValueError("Invalid key")

class SCM_binary(DataSCM):
    def __init__(self, config):
        super().__init__(config)

    def propensity_full(self, a, x, u_m1, u_m2, u_y):
        score = 3 * np.mean(x, axis=1, keepdims=True) + self.config["coef_u_m1"] * u_m1 \
                + self.config["coef_u_m2"] * u_m2 + self.config["coef_u_y"] * u_y
        p_a1_x = self.sigmoid(score)
        # Prohibit overlap violations
        p_a1_x = np.clip(p_a1_x, 0.05, 0.95)
        if a == 0:
            return 1 - p_a1_x
        else:
            return p_a1_x
        #p_a0_x = 1 - p_a1_x
        #p_x = np.concatenate((p_a0_x, p_a1_x), axis=1)
        #p_a_x = p_x[np.arange(p_x.shape[0]), a[:, 0].astype(int)]
        #return np.expand_dims(p_a_x, axis=1)

    def get_gamma_maxratio(self, maxratio, prop_obs):
        return (maxratio * (prop_obs - 1)) / (maxratio * prop_obs - 1)

    def get_gamma_minratio(self, minratio, prop_obs):
        return (1 - minratio * prop_obs) / (minratio * (1 - prop_obs))

    def f_a(self, x, u_m1, u_m2, u_y):
        # Observed propensity score
        prop = self.propensity_full(a=1, x=x, u_m1=u_m1, u_m2=u_m2, u_y=u_y)
        return np.random.binomial(1, prop)

    def f_m1(self, x, a, u_m1, eps_m1):
        noise_term = self.config["noise_m1"] * ((u_m1 - 0.5) + eps_m1)
        score = a * (np.sin(1 * x)) + (1 - a) * (np.sin(4 * x)) + noise_term
        return np.where(score > 0, 1, 0)

    def f_m2(self, x, a, m1, u_m2, eps_m2):
        noise_term = self.config["noise_m2"] * ((u_m2 - 0.5) + eps_m2)
        score = a * m1 * (np.sin(1 * x)) + (1 - a) * m1 * (np.sin(4 * x))
        score += -a * (1 - m1) * (np.sin(1 * x)) - (1 - a) * (1 - m1) * (np.sin(4 * x)) + noise_term
        return np.where(score > 0, 1, 0)

    def f_y(self, x, a, m1, m2, u_y, eps_y):
        noise_term = self.config["noise_y"] * ((u_y - 0.5) + eps_y)
        score = a * m1 * m2 * (np.sin(1 * x)) + (1 - a) * m1 * m2 * (np.sin(4 * x))
        score += a * m1 * (1 - m2) * (np.sin(8 * x)) + (1 - a) * m1 * (1 - m2) * (np.sin(1 * x))
        score += -a * (1 - m1) * m2 * (np.sin(1 * x)) - (1 - a) * (1 - m1) * m2 * (np.sin(4 * x))
        score += -a * (1 - m1) * (1 - m2) * (np.sin(8 * x)) - (1 - a) * (1 - m1) * (1 - m2) * (
            np.sin(1 * x)) + noise_term
        return score


class SCM_continuous(DataSCM):
    def __init__(self, config):
        super().__init__(config)

    def get_par_beta(self, x, u_m1, u_m2, u_y):
        par1 = np.mean(x, axis=1, keepdims=True) + 2 + self.config["coef_u_m1"] * (u_m1 - 0.5) \
                + self.config["coef_u_m2"] * (u_m2 - 0.5) + self.config["coef_u_y"] * (u_y - 0.5)
        par2 = np.mean(x, axis=1, keepdims=True) + 2 + self.config["coef_u_m1"] * (u_m1 - 0.5) \
                + self.config["coef_u_m2"] * (u_m2 - 0.5) + self.config["coef_u_y"] * (u_y - 0.5)
        par1 = np.clip(par1, 0.1, 10)
        par2 = np.clip(par2, 0.1, 10)
        return par1, par2

    def propensity_full(self, a, x, u_m1, u_m2, u_y):
        #Parameter of beta distribution (par1=a, par2=b)
        par1, par2 = self.get_par_beta(x, u_m1, u_m2, u_y)
        prob = beta.pdf(a, par1, par2)
        return prob

    def get_gamma_maxratio(self, maxratio, prop_obs):
        return maxratio

    def get_gamma_minratio(self, minratio, prop_obs):
        return 1 / minratio

    def f_a(self, x, u_m1, u_m2, u_y):
        par1, par2 = self.get_par_beta(x, u_m1, u_m2, u_y)
        return beta.rvs(par1, par2)

    def f_m1(self, x, a, u_m1, eps_m1):
        noise_term = self.config["noise_m1"] * ((u_m1 - 0.5) + eps_m1)
        score = a * (np.sin(1 * x)) + (1 - a) * (np.sin(4 * x)) + noise_term
        return np.where(score > 0, 1, 0)

    def f_m2(self, x, a, m1, u_m2, eps_m2):
        noise_term = self.config["noise_m2"] * ((u_m2 - 0.5) + eps_m2)
        score = a * m1 * (np.sin(1 * x)) + (1 - a) * m1 * (np.sin(4 * x))
        score += -a * (1 - m1) * (np.sin(1 * x)) - (1 - a) * (1 - m1) * (np.sin(4 * x)) + noise_term
        return np.where(score > 0, 1, 0)

    def f_y(self, x, a, m1, m2, u_y, eps_y):
        noise_term = self.config["noise_y"] * ((u_y - 0.5) + eps_y)
        score = a * m1 * m2 * (np.sin(1 * x)) + (1 - a) * m1 * m2 * (np.sin(4 * x))
        score += a * m1 * (1 - m2) * (np.sin(8 * x)) + (1 - a) * m1 * (1 - m2) * (np.sin(1 * x))
        score += -a * (1 - m1) * m2 * (np.sin(1 * x)) - (1 - a) * (1 - m1) * m2 * (np.sin(4 * x))
        score += -a * (1 - m1) * (1 - m2) * (np.sin(8 * x)) - (1 - a) * (1 - m1) * (1 - m2) * (
            np.sin(1 * x)) + noise_term
        return score


class SCM_continuous_weight(SCM_continuous):
    def __init__(self, config):
        super().__init__(config)

    def get_par_beta(self, x, u_m1, u_m2, u_y):
        par1 = np.mean(x, axis=1, keepdims=True) + np.where(np.mean(x, axis=1, keepdims=True) < 0, 1, 0) * (2 + self.config["coef_u_m1"] * (u_m1 - 0.5) \
                + self.config["coef_u_m2"] * (u_m2 - 0.5) + self.config["coef_u_y"] * (u_y - 0.5))
        par2 = np.mean(x, axis=1, keepdims=True) + 2 + np.where(np.mean(x, axis=1, keepdims=True) < 0, 1, 0) * (self.config["coef_u_m1"] * (u_m1 - 0.5) \
                + self.config["coef_u_m2"] * (u_m2 - 0.5) + self.config["coef_u_y"] * (u_y - 0.5))
        par1 = np.clip(par1, 0.1, 10)
        par2 = np.clip(par2, 0.1, 10)
        return par1, par2