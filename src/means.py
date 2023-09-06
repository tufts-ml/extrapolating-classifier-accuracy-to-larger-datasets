import numpy as np
import torch
import gpytorch
# Importing our custom module(s)
import priors

class PowerLawPriorMean(gpytorch.means.Mean):
    def __init__(self, y_max, epsilon_min=0.0):
        super().__init__()
        assert y_max >= 0.0 and y_max <= 1.0, 'y_max is less than 0.0 or greater than 1.0'
        self.y_max = y_max
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor(0.0))
        self.theta1 = torch.nn.Parameter(torch.tensor(0.0))
        self.theta2 = torch.nn.Parameter(torch.tensor(0.0))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        epsilon = self.epsilon_min + (1.0-self.y_max-self.epsilon_min)*torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)        
        return (1.0 - epsilon) - (theta1 * torch.pow(x.ravel(), theta2))
    
class ArctanPriorMean(gpytorch.means.Mean):
    def __init__(self, y_max, epsilon_min=0.0):
        super().__init__()
        assert y_max >= 0.0 and y_max <= 1.0, 'y_max is less than 0.0 or greater than 1.0'
        self.y_max = y_max
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor(0.0))
        self.theta1 = torch.nn.Parameter(torch.tensor(0.0))
        self.theta2 = torch.nn.Parameter(torch.tensor(0.0))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        epsilon = self.epsilon_min + (1.0-self.y_max-self.epsilon_min)*torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        return 2/np.pi * torch.atan(theta1 * np.pi/2 * x.ravel() + theta2) - epsilon