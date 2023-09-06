import numpy as np
import scipy
import torch
import gpytorch

def uniform_probability_mass(y, loc, scale, epsilon=0.01):
    probability_lower = scipy.stats.uniform.cdf(y-epsilon, loc=loc, scale=scale)
    probability_upper = scipy.stats.uniform.cdf(y+epsilon, loc=loc, scale=scale)
    probability_range = probability_upper - probability_lower
    return np.mean(probability_range)

def truncnorm_probability_mass(y, a, b, loc, scale, epsilon=0.01):
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    lower_probability = scipy.stats.truncnorm.cdf(y-epsilon, alpha, beta, loc=loc, scale=scale)
    upper_probability = scipy.stats.truncnorm.cdf(y+epsilon, alpha, beta, loc=loc, scale=scale)
    probability_mass = upper_probability - lower_probability
    return np.mean(probability_mass)

def truncated_normal_uncertainty(a, b, loc, scale, lower_percentile=0.025, upper_percentile=0.975):
    # Calculate the upper and lower quantiles of the truncated normal distribution
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    lower_quantile = scipy.stats.truncnorm(alpha, beta, loc=loc, scale=scale).ppf(lower_percentile)
    upper_quantile = scipy.stats.truncnorm(alpha, beta, loc=loc, scale=scale).ppf(upper_percentile)

    # Clip the quantiles to ensure they remain within the truncation bounds
    lower_quantile = np.maximum(lower_quantile, a)
    upper_quantile = np.minimum(upper_quantile, b)

    return lower_quantile, upper_quantile

def my_truncnorm(a, b, loc, scale):
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    return scipy.stats.truncnorm(alpha, beta, loc, scale)

def calc_outputscale_prior(tau_prior, desired_low, desired_high, num_samples=1000):
    desired_medium = (desired_high+desired_low)/2
    best_dist = np.inf
    best_m, best_s = 0, 0
    tau_samples = tau_prior.rvs(num_samples)
    for m in np.linspace(0.001, 0.05, 100):
        for s in np.linspace(0.001, 0.05, 100):
            outputscale_prior = my_truncnorm(0.0, np.inf, m, s)
            outputscale_samples = outputscale_prior.rvs(num_samples)
            expression_values = 6*np.sqrt(outputscale_samples**2+tau_samples**2)
            low, medium, high = np.percentile(expression_values, [20, 50, 80])
            if abs(desired_low - low) + abs(desired_medium - medium) + abs(desired_high - high) < best_dist:
                best_dist = abs(desired_low - low) + abs(desired_medium - medium) + abs(desired_high - high)
                best_m, best_s = m, s
    outputscale_prior = my_truncnorm(0.0, np.inf, best_m, best_s)
    outputscale_samples = outputscale_prior.rvs(num_samples)
    expression_values = 6*np.sqrt(outputscale_samples**2+tau_samples**2)
    return best_m, best_s
    
class UniformPrior(gpytorch.priors.Prior):   
    arg_constraints = {} # Please set `arg_constraints = {}` or initialize the distribution with `validate_args=False` to turn off validation.
    def __init__(self, a, b):
        super(UniformPrior, self).__init__()
        self.a = a
        self.b = b
        
    def forward(self, x):
        return torch.tensor(1/abs(self.a - self.b)) if (x >= self.a and x <= self.b) else torch.tensor(0.0)
        
    def log_prob(self, x):
        return torch.log(self.forward(x))

class TruncatedNormalPrior(gpytorch.priors.Prior):
    arg_constraints = {} # Please set `arg_constraints = {}` or initialize the distribution with `validate_args=False` to turn off validation.
    def __init__(self, a, b, loc, scale):
        super(TruncatedNormalPrior, self).__init__()
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        
    def log_prob(self, x):
        alpha = torch.tensor((self.a - self.loc) / self.scale)
        beta = torch.tensor((self.b - self.loc) / self.scale)
        Phi_alpha = 1/2*(1+torch.erf(alpha/np.sqrt(2))) if not self.a == -np.inf else 0
        Phi_beta = 1/2*(1+torch.erf(beta/np.sqrt(2))) if not self.b == np.inf else 1
        log_phi_x = np.log(1/np.sqrt(2*np.pi))+(-1/2*((x-self.loc)/self.scale)**2)
        return np.log(1/self.scale)+log_phi_x-np.log(Phi_beta-Phi_alpha+1e-15) if (x >= self.a and x <= self.b) else -torch.inf    
    def forward(self, x):
        alpha = torch.tensor((self.a - self.loc) / self.scale)
        beta = torch.tensor((self.b - self.loc) / self.scale)
        Phi_alpha = 1/2*(1+torch.erf(alpha/np.sqrt(2))) if not self.a == -np.inf else 0
        Phi_beta = 1/2*(1+torch.erf(beta/np.sqrt(2))) if not self.b == np.inf else 1
        phi_x = (1/np.sqrt(2*np.pi))*np.exp(-1/2*((x-self.loc)/self.scale)**2)
        return (1/self.scale)*phi_x/(Phi_beta-Phi_alpha+1e-15) if (x >= self.a and x <= self.b) else torch.tensor(0.0)