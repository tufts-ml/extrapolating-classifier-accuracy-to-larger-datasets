import torch
import gpytorch
import torch.nn as nn
import numpy as np

from priors import *

class PowerLaw(nn.Module):
    def __init__(self, max_y):
        super().__init__()
        assert max_y >= 0.0 and max_y <= 1.0, 'max_x is less than 0.0 or greater than 1.0'
        self.max_y = max_y
        self.one_minus_epsilon = torch.nn.Parameter(torch.tensor([1.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        one_minus_epsilon = self.max_y + (1-self.max_y)*torch.sigmoid(self.one_minus_epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)
        return one_minus_epsilon - (theta1 * torch.pow(x, theta2)).ravel()
    
class Arctan(nn.Module):
    def __init__(self, max_y):
        super().__init__()
        assert max_y >= 0.0 and max_y <= 1.0, 'max_x is less than 0.0 or greater than 1.0'
        self.max_y = max_y
        self.one_minus_epsilon = torch.nn.Parameter(torch.tensor([1.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta3 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        one_minus_epsilon = self.max_y + (1-self.max_y)*torch.sigmoid(self.one_minus_epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        theta3 = -self.softplus(-self.theta3)
        return 2/np.pi * (-theta3 + one_minus_epsilon) * torch.atan(theta1 * np.pi/2 * x + theta2).ravel() + theta3

class PowerLawPriorMean(gpytorch.means.Mean):
    def __init__(self, max_y):
        super().__init__()
        assert max_y >= 0.0 and max_y <= 1.0, 'max_x is less than 0.0 or greater than 1.0'
        self.max_y = max_y
        self.one_minus_epsilon = torch.nn.Parameter(torch.tensor([1.0]))
        self.theta1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.theta2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.softplus = torch.nn.Softplus()
        self.register_prior('one_minus_epsilon_prior', EpsilonPrior(self.max_y, 1.0), 
                            lambda module: module.max_y+(1-module.max_y)*torch.sigmoid(module.one_minus_epsilon))

    def forward(self, x):
        one_minus_epsilon = self.max_y + (1-self.max_y)*torch.sigmoid(self.one_minus_epsilon)
        one_minus_epsilon = 1.0
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)        
        return one_minus_epsilon - (theta1 * torch.pow(x, theta2)).ravel()
    
class GPPowerLaw(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super(GPPowerLaw, self).__init__(X, y, likelihood)
        self.mean_module = PowerLawPriorMean(torch.max(y))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior = LengthscalePrior()),
            outputscale_prior = OutputscalePrior()
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
        self.register_prior('one_minus_epsilon_prior', EpsilonPrior(self.max_y, 1.0),
                            lambda module: module.max_y+(1-module.max_y)*torch.sigmoid(module.one_minus_epsilon))

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
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior = LengthscalePrior()),
            outputscale_prior = OutputscalePrior()
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def split_df(df, index):
    X_train = torch.Tensor(df[df.n<=360]['n'].to_numpy())
    y_train = torch.Tensor(np.array(df[df.n<=360]['test_auroc'].to_list())[:,index])
    X_test = torch.Tensor(df[df.n>360]['n'].to_numpy())
    y_test = torch.Tensor(np.array(df[df.n>360]['test_auroc'].to_list())[:,index])
    return X_train, y_train, X_test, y_test

def train_PowerLaw(X, y, training_iter=100000):
    model = PowerLaw(torch.max(y))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
    #for name, param in model.named_parameters():
    #    print(name, param)
    return model

def train_GPPowerLaw(X, y, training_iter=10000):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=TauPrior())
    likelihood.train()
    model = GPPowerLaw(X, y, likelihood)
    #model.covar_module.outputscale = 0.0009
    #model.covar_module.base_kernel.lengthscale = 2.0
    model.train()
    #parameters = [param for name, param in model.named_parameters() if name not in ['covar_module.raw_outputscale', 'covar_module.base_kernel.raw_lengthscale']]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
    #for name, param in model.named_parameters():
    #    print(name, param)
    return likelihood, model

def train_GPArctan(X, y, training_iter=10000):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=TauPrior())
    likelihood.train()
    model = GPArctan(X, y, likelihood)
    model.train()    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
    #for name, param in model.named_parameters():
    #    print(name, param)
    return likelihood, model

def train_Arctan(X, y, training_iter=100000):
    model = Arctan(torch.max(y))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
    #for name, param in model.named_parameters():
    #    print(name, param)
    return model