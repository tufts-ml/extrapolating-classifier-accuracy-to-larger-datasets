import torch
import gpytorch
import torch.nn as nn
import numpy as np

from priors import *

class PowerLaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 0.0
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)
        return (1.0 - self.epsilon) - (theta1 * torch.pow(x, theta2)).ravel()
    
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

class PowerLawPriorMean(gpytorch.means.Mean):
    def __init__(self, max_y):
        super().__init__()
        assert max_y >= 0.0 and max_y <= 1.0, 'max_x is less than 0.0 or greater than 1.0'
        self.max_y = max_y
        self.one_minus_epsilon = torch.nn.Parameter(torch.tensor([1.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()
        self.register_prior(
            'one_minus_epsilon_prior',
            TrianglePrior(self.max_y, 1.0),
            lambda module: module.one_minus_epsilon,
        )

    def forward(self, x):
        one_minus_epsilon = self.max_y + (1-self.max_y)*torch.sigmoid(self.one_minus_epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)        
        return one_minus_epsilon - (theta1 * torch.pow(x, theta2)).ravel()
    
class GPPowerLaw(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super(GPPowerLaw, self).__init__(X, y, likelihood)
        self.mean_module = PowerLawPriorMean(torch.max(y))
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior = gpytorch.priors.NormalPrior(50, 5)
            ),
            outputscale_prior = gpytorch.priors.LogNormalPrior(0, np.exp(1/30))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ArctanPriorMean(gpytorch.means.Mean):
    def __init__(self, max_y):
        super().__init__()
        assert max_y >= 0.0 and max_y <= 1.0, 'max_x is less than 0.0 or greater than 1.0'
        self.max_y = max_y
        self.one_minus_epsilon = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta3 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()
        self.register_prior(
            'one_minus_epsilon_prior',
            TrianglePrior(self.max_y, 1.0),
            lambda module: module.one_minus_epsilon,
        )

    def forward(self, x):
        one_minus_epsilon = self.max_y + (1-self.max_y)*torch.sigmoid(self.one_minus_epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        theta3 = -self.softplus(-self.theta3)
        return 2/np.pi * (-theta3 + one_minus_epsilon) * torch.atan(theta1 * np.pi/2 * x + theta2).ravel() + theta3

class GPArctan(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super(GPArctan, self).__init__(X, y, likelihood)
        self.mean_module = ArctanPriorMean(torch.max(y))
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior = gpytorch.priors.LogNormalPrior(50, 1)
            ),
            outputscale_prior = gpytorch.priors.LogNormalPrior(0, np.exp(1/30))
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)