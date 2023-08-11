import numpy as np
import torch
import torch.nn as nn
import gpytorch

from priors import *

class PowerLaw(nn.Module):
    def __init__(self, y_n, epsilon_min=0.0):
        super().__init__()
        assert y_n >= 0.0 and y_n <= 1.0, 'y_n is less than 0.0 or greater than 1.0'
        self.y_n = y_n
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        epsilon = self.epsilon_min + (1.0-self.y_n-self.epsilon_min)*torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)
        return (1.0 - epsilon) - (theta1 * torch.pow(x, theta2)).ravel()
    
class Arctan(nn.Module):
    def __init__(self, y_n, epsilon_min=0.0):
        super().__init__()
        assert y_n >= 0.0 and y_n <= 1.0, 'y_n is less than 0.0 or greater than 1.0'
        self.y_n = y_n
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta3 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        epsilon = self.epsilon_min + (1.0-self.y_n-self.epsilon_min)*torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        theta3 = -self.softplus(-self.theta3)
        return 2/np.pi * (-theta3 + (1.0 - epsilon)) * torch.atan(theta1 * np.pi/2 * x + theta2).ravel() + theta3

class PowerLawPriorMean(gpytorch.means.Mean):
    def __init__(self, y_n, epsilon_min=0.0):
        super().__init__()
        assert y_n >= 0.0 and y_n <= 1.0, 'y_n is less than 0.0 or greater than 1.0'
        self.y_n = y_n
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()
        self.register_prior(
            'epsilon_prior', 
            EpsilonPrior((1.0-self.y_n), self.epsilon_min),
            lambda module: module.epsilon_min + (1.0-module.y_n-module.epsilon_min)*torch.sigmoid(module.epsilon)
        )

    def forward(self, x):
        epsilon = self.epsilon_min + (1.0-self.y_n-self.epsilon_min)*torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)        
        return (1.0 - epsilon) - (theta1 * torch.pow(x, theta2)).ravel()
    
class GPPowerLaw(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, epsilon_min=0.0):
        super(GPPowerLaw, self).__init__(X, y, likelihood)
        # Registers a prior on the sqrt of the noise parameter
        # (e.g., a prior for the noise standard deviation instead of variance)
        self.register_prior(
            'noise_std_prior', 
            TruncatedNormalPrior(0, np.inf, 0.01, 0.005), 
            lambda module: module.likelihood.noise_covar.noise.sqrt()
        )
        # Calculate optimal outputscale prior
        tau_prior = my_truncnorm(0, np.inf, 0.01, 0.005)
        desired_low = (1/2)*((1-epsilon_min)-y[-1].item())
        desired_high = (3/4)*((1-epsilon_min)-y[-1].item())
        m, s = calc_outputscale_prior(tau_prior, desired_low, desired_high)
        # Mean module
        self.mean_module = PowerLawPriorMean(y[-1].item(), epsilon_min)
        # Covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior = TruncatedNormalPrior(0, np.inf, -1.23, 2.12)),
        )
        self.register_prior(
            'outputscale_std_prior', 
            TruncatedNormalPrior(0, np.inf, m, s), 
            lambda module: module.covar_module.outputscale.sqrt()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ArctanPriorMean(gpytorch.means.Mean):
    def __init__(self, y_n, epsilon_min=0.0):
        super().__init__()
        assert y_n >= 0.0 and y_n <= 1.0, 'y_n is less than 0.0 or greater than 1.0'
        self.y_n = y_n
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta3 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()
        self.register_prior(
            'epsilon_prior',
            EpsilonPrior((1.0-self.y_n), self.epsilon_min),
            lambda module: module.epsilon_min + (1.0-module.y_n-module.epsilon_min)*torch.sigmoid(module.epsilon)
        )

    def forward(self, x):
        epsilon = self.epsilon_min + (1.0-self.y_n-self.epsilon_min)*torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        theta3 = -self.softplus(-self.theta3)
        return 2/np.pi * (-theta3 + (1.0 - epsilon)) * torch.atan(theta1 * np.pi/2 * x + theta2).ravel() + theta3

class GPArctan(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, epsilon_min=0.0):
        super(GPArctan, self).__init__(X, y, likelihood)
        # Registers a prior on the sqrt of the noise parameter
        # (e.g., a prior for the noise standard deviation instead of variance)
        self.register_prior(
            'noise_std_prior', 
            TruncatedNormalPrior(0, np.inf, 0.01, 0.005), 
            lambda module: module.likelihood.noise_covar.noise.sqrt()
        )
        # Calculate optimal outputscale prior
        tau_prior = my_truncnorm(0, np.inf, 0.01, 0.005)
        desired_low = (1/2)*((1-epsilon_min)-y[-1].item())
        desired_high = (3/4)*((1-epsilon_min)-y[-1].item())
        m, s = calc_outputscale_prior(tau_prior, desired_low, desired_high)
        # Mean module
        self.mean_module = ArctanPriorMean(y[-1].item(), epsilon_min)
        # Covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior = TruncatedNormalPrior(0, np.inf, -1.23, 2.12)),
        )
        self.register_prior(
            'outputscale_std_prior', 
            TruncatedNormalPrior(0, np.inf, m, s), 
            lambda module: module.covar_module.outputscale.sqrt()
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_PowerLaw(X, y, lr=0.001, training_iter=100000):
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PowerLaw(y[-1].item())
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss
    if device.type == "cuda": model.to('cpu')
    return model, losses

def train_Arctan(X, y, lr=0.01, training_iter=100000):
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Arctan(y[-1].item())
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss
    if device.type == "cuda": model.to('cpu')
    return model, losses

def train_GPPowerLaw(X, y, lr=0.01, training_iter=50000):
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.train()
    model = GPPowerLaw(X, y, likelihood)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss
    #print('Variance: {}'.format(6*np.sqrt(model.covar_module.outputscale.item() + model.likelihood.noise_covar.noise.item())))
    #print('Tau: {}'.format(np.sqrt(model.likelihood.noise_covar.noise.item())))
    #print('Outputscale: {}'.format(np.sqrt(model.covar_module.outputscale.item())))
    #print('Lengthscale: {}'.format(model.covar_module.base_kernel.lengthscale.item()))
    if device.type == "cuda": model.to('cpu')
    return likelihood, model, losses

def train_GPArctan(X, y, lr=0.01, training_iter=50000):
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.train()
    model = GPArctan(X, y, likelihood)
    model.to(device)
    model.train()    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss    
    #print('Variance: {}'.format(6*np.sqrt(model.covar_module.outputscale.item() + model.likelihood.noise_covar.noise.item())))
    #print('Tau: {}'.format(np.sqrt(model.likelihood.noise_covar.noise.item())))
    #print('Outputscale: {}'.format(np.sqrt(model.covar_module.outputscale.item())))
    #print('Lengthscale: {}'.format(model.covar_module.base_kernel.lengthscale.item()))
    if device.type == "cuda": model.to('cpu')
    return likelihood, model, losses