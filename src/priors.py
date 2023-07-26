import numpy as np
import scipy
import torch
import gpytorch

def truncated_normal_uncertainty(mu_predicted, sigma_predicted, lower_bound=0.0, upper_bound=1.0):
    # Calculate the upper and lower quantiles of the truncated normal distribution
    lower_quantile = scipy.stats.truncnorm.ppf(0.025, (lower_bound - mu_predicted) / sigma_predicted, (upper_bound - mu_predicted) / sigma_predicted, loc=mu_predicted, scale=sigma_predicted)
    upper_quantile = scipy.stats.truncnorm.ppf(0.975, (lower_bound - mu_predicted) / sigma_predicted, (upper_bound - mu_predicted) / sigma_predicted, loc=mu_predicted, scale=sigma_predicted)

    # Clip the quantiles to ensure they remain within the truncation bounds
    lower_quantile = np.maximum(lower_quantile, lower_bound)
    upper_quantile = np.minimum(upper_quantile, upper_bound)

    return lower_quantile, upper_quantile

def uniform_likelihood(values, lower=0.0, upper=1.0):
    likelihood = 0
    for value in values:
        if value < lower or value > upper:
            likelihood += -np.inf
        else:
            likelihood += np.log(1 / (upper - lower))
    return np.sum(likelihood)

def truncated_normal_likelihood(y, mu, sigma, lower_bound=0.0, upper_bound=1.0):
    # Calculate the log-likelihood using the truncated normal distribution
    likelihood = scipy.stats.truncnorm.logpdf(y, (lower_bound - mu) / sigma, (upper_bound - mu) / sigma, loc=mu, scale=sigma)
    
    return np.sum(likelihood)

class TauPrior(gpytorch.priors.Prior):    
    def __init__(self):
        super(TauPrior, self).__init__()
        self.mu = 0.01
        self.sigma = 1/30
        self.lower_bound = 0.0
        self.upper_bound = np.inf
        
    def log_prob(self, x):
        a = torch.tensor((self.lower_bound - self.mu) / self.sigma)
        b = torch.tensor((self.upper_bound - self.mu) / self.sigma)
        log_prob = (-0.5 * ((x - self.mu) / self.sigma) ** 2) - np.log(self.sigma * np.sqrt(2 * np.pi)) - torch.log(torch.erf(b / np.sqrt(2)) - torch.erf(a / np.sqrt(2)) + 1e-15)
        return log_prob
    
class OutputscalePrior(gpytorch.priors.Prior):    
    def __init__(self):
        super(OutputscalePrior, self).__init__()
        self.mu = 1/30
        self.sigma = 0.01
        self.lower_bound = 0.0
        self.upper_bound = np.inf
        
    def log_prob(self, x):
        a = torch.tensor((self.lower_bound - self.mu) / self.sigma)
        b = torch.tensor((self.upper_bound - self.mu) / self.sigma)
        log_prob = (-0.5 * ((x - self.mu) / self.sigma) ** 2) - np.log(self.sigma * np.sqrt(2 * np.pi)) - torch.log(torch.erf(b / np.sqrt(2)) - torch.erf(a / np.sqrt(2)) + 1e-15)
        return log_prob    
    
class LengthscalePrior(gpytorch.priors.Prior):    
    def __init__(self):
        super(LengthscalePrior, self).__init__()
        self.mu = 6
        self.sigma = 3
        self.lower_bound = 0.0
        self.upper_bound = np.inf
        
    def log_prob(self, x):
        a = torch.tensor((self.lower_bound - self.mu) / self.sigma)
        b = torch.tensor((self.upper_bound - self.mu) / self.sigma)
        log_prob = (-0.5 * ((x - self.mu) / self.sigma) ** 2) - np.log(self.sigma * np.sqrt(2 * np.pi)) - torch.log(torch.erf(b / np.sqrt(2)) - torch.erf(a / np.sqrt(2)) + 1e-15)
        return log_prob
    
class EpsilonPrior(gpytorch.priors.Prior):    
    def __init__(self, min_val, max_val):
        super(EpsilonPrior, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.slope = (2/self.max_val - self.min_val)/(self.max_val - self.min_val)
        self.intercept = 2 - self.slope*self.max_val
        
    def log_prob(self, x):
        return torch.log(self.slope*x+self.intercept) if (x <= self.max_val and x >= self.min_val) else torch.Tensor([-np.inf])