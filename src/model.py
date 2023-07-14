import scipy
import torch
import gpytorch
import torch.nn as nn
import numpy as np

class PowerLaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 0.0
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        theta1 = self.softplus(self.theta1)
        theta2 = torch.sigmoid(self.theta2)
        return (1.0 - self.epsilon) - (theta1 * torch.pow(x, -theta2)).ravel()
    
class Arctan(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 0.0
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta3 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        theta3 = -self.softplus(-self.theta3)
        return 2/np.pi * (-theta3 + (1.0 - self.epsilon)) * torch.atan(theta1 * np.pi/2 * x + theta2).ravel() + theta3

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

class PowerLawPriorMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        self.epsilon = 0.0
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        theta1 = self.softplus(self.theta1)
        theta2 = torch.sigmoid(self.theta2)        
        return (1.0 - self.epsilon) - (theta1 * torch.pow(x, -theta2)).ravel()
    
class GPPowerLaw(gpytorch.models.ExactGP):
    def __init__(self, n_list, metric_list, likelihood):
        super(GPPowerLaw, self).__init__(n_list, metric_list, likelihood)
        self.mean_module = PowerLawPriorMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
            lengthscale_prior = gpytorch.priors.NormalPrior(1000, 1e-6),
            outputscale_prior = gpytorch.priors.LogNormalPrior(0, 1)
        )
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ArctanPriorMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        self.epsilon = 0.0
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta3 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        theta3 = -self.softplus(-self.theta3)
        return 2/np.pi * (-theta3 + (1.0 - self.epsilon)) * torch.atan(theta1 * np.pi/2 * x + theta2).ravel() + theta3

class GPArctan(gpytorch.models.ExactGP):
    def __init__(self, n_list, metric_list, likelihood):
        super(GPArctan, self).__init__(n_list, metric_list, likelihood)
        self.mean_module = ArctanPriorMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
            lengthscale_prior = gpytorch.priors.NormalPrior(1000, 1e-6),
            outputscale_prior = gpytorch.priors.LogNormalPrior(0, 1)
        )
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)