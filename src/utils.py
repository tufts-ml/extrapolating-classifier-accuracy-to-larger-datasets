import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
# Importing our custom module(s)
import metrics
import models
import priors

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_experiment(path):
    df = pd.read_csv(path, index_col='Unnamed: 0')
    for column in ['train_BA', 'train_auroc', 'val_BA', 'val_auroc', 'test_BA', 'test_auroc']:
        df[column] = df[column].apply(lambda item: np.fromstring(item[1:-1], sep=' '))
    return df

def split_df(df, index):
    X_train = torch.Tensor(df[df.n<=360]['n'].to_numpy())
    y_train = torch.Tensor(np.array(df[df.n<=360]['test_auroc'].to_list())[:,index])
    X_test = torch.Tensor(df[df.n>360]['n'].to_numpy())
    y_test = torch.Tensor(np.array(df[df.n>360]['test_auroc'].to_list())[:,index])
    return X_train, y_train, X_test, y_test

def load_dataset(df):
    X = torch.vstack([torch.load(path).float() for path in df.path.to_list()]).detach().numpy()
    y = torch.tensor(df.label.to_list()).detach().numpy()
    return X, y

class EncodedDataset(Dataset):
    def __init__(self, df, max_slices=50):
        self.max_slices = max_slices
        self.path = df.path.to_list()
        self.image = [self.transform(torch.load(path, map_location='cpu').float().detach().numpy()) for path in self.path]
        self.label = df.label.to_list()
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        return self.image[index], self.label[index]
    
    def transform(self, image):
        slices = min(self.max_slices, image.shape[0])
        linspace = np.linspace(start=0, stop=image.shape[0]-1, num=slices, dtype=int)
        return image[linspace]
        
def collate_fn(batch):
    images, labels = zip(*batch)
    images = np.concatenate(images, axis=0)
    labels = np.array(labels)
    return torch.Tensor(images), tuple([image.shape[0] for image, label in batch]), torch.Tensor(labels)

def train_one_epoch(model, device, optimizer, loss_func, data_loader, args=None):

    model.train()

    running_loss = 0.0
    label_list, prediction_list = list(), list()

    for i, data in enumerate(data_loader, 0):
        inputs, slices, labels = data
        
        if device.type == 'cuda':
            inputs = inputs.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, slices)
        loss = (1/np.log(2))*(len(slices)/len(data_loader.dataset))*loss_func(outputs, labels)
        #loss = (1/np.log(2))*(len(slices)/len(data_loader.dataset))*loss_func(outputs, labels, model)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        if device.type == 'cuda':
            loss = loss.cpu()
        
        running_loss += loss.detach().numpy()
        optimizer.step()
        
    return running_loss

def evaluate(model, device, loss_func, data_loader):
    
    model.eval()

    with torch.no_grad():

        running_loss = 0.0
        label_list, prediction_list = list(), list()

        for i, data in enumerate(data_loader, 0):
            inputs, slices, labels = data
                        
            if device.type == 'cuda':
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = model(inputs, slices)
            loss = (1/np.log(2))*(len(slices)/len(data_loader.dataset))*loss_func(outputs, labels)
            #loss = (1/np.log(2))*(len(slices)/len(data_loader.dataset))*loss_func(outputs, labels, model)

            if device.type == 'cuda':
                loss = loss.cpu()
                labels = labels.cpu()
                outputs = outputs.cpu()

            for output, label in zip(outputs.cpu(), labels):
                label_list.append(label.numpy().astype(int))
                prediction_list.append(output.numpy())

            running_loss += loss.numpy()
            
    return running_loss, label_list, prediction_list

# TODO: Should I create a seperate file for plotting helper functions?
def print_metrics(model_objects, y_train, X_test, y_test, verbose=1):
    model, *likelihood_objects = model_objects
    label_map = { models.PowerLaw: 'Power law', models.Arctan: 'Arctan', models.GPPowerLaw: 'GP pow', models.GPArctan: 'GP arc' }
    label = label_map.get(type(model), 'Unknown') # Default label is 'Unknown' 
    if label.startswith('GP'):
        likelihood, = likelihood_objects
        with torch.no_grad(): predictions = likelihood(model(X_test))
        loc = predictions.mean.numpy()
        scale = predictions.stddev.numpy()
        lower, upper = priors.truncated_normal_uncertainty(0.0, 1.0, loc, scale)   
        error = metrics.rmse(y_test.detach().numpy(), loc)
        upm = priors.uniform_probability_mass(y_test.detach().numpy(), torch.min(y_train).item(), 1-torch.min(y_train).item())
        tpm = priors.truncnorm_probability_mass(y_test.detach().numpy(), 0.0, 1.0, loc, scale)
        coverage_95 = metrics.coverage(y_test.detach().numpy(), lower, upper)
        lower, upper = priors.truncated_normal_uncertainty(0.0, 1.0, loc, scale, 0.1, 0.9)
        coverage_80 = metrics.coverage(y_test.detach().numpy(), lower, upper)
        if verbose:
            print('{} RMSE: {:.4f}'.format(label, 100*error))
            print('Uniform probability mass: {:.4f}'.format(upm))
            print('{} probability mass: {:.4f}'.format(label, tpm))
            print('{} 80% coverage: {:.2f}%'.format(label, 100*coverage_80))      
            print('{} 95% coverage: {:.2f}%\n'.format(label, 100*coverage_95))
        return 100*error, upm, tpm, 100*coverage_80, 100*coverage_95
    else:
        with torch.no_grad(): loc = model(X_test).detach().numpy()
        error = metrics.rmse(y_test.detach().numpy(), loc)
        if verbose:
            print('{} RMSE: {:.4f}\n'.format(label, 100*error))
        return 100*error,
    
def plot_data(ax, X_train, y_train, X_test, y_test):
    ax.scatter(X_train, y_train, color='black', alpha=1.0, label='Initial subsets')
    ax.scatter(X_test, y_test, color='black', alpha=0.3, label='Ground truth')

def load_model(name, path, X_train, y_train):
    model_map = { 'PowerLaw': models.PowerLaw, 'Arctan': models.Arctan, 'GPPowerLaw': models.GPPowerLaw, 'GPArctan': models.GPArctan }
    model_class = model_map[name]
    if name.startswith('GP'):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = model_class(X_train, y_train, likelihood)
        model.load_state_dict(torch.load(path))
        likelihood.eval()
        model.eval()
        return model, likelihood
    else:
        model = model_class(y_train[-1].item())
        model.load_state_dict(torch.load(path))
        model.eval()
        return model,

def plot_model(ax, model_objects, color='black'):
    model, *likelihood_objects = model_objects
    label_map = { models.PowerLaw: 'Power law', models.Arctan: 'Arctan', models.GPPowerLaw: 'GP pow (ours)', models.GPArctan: 'GP arc (ours)' }
    label = label_map.get(type(model), 'Unknown') # Default label is 'Unknown' 
    if label.startswith('GP'):
        likelihood, = likelihood_objects
        linspace = torch.linspace(50, 30000, 29950)
        with torch.no_grad(): predictions = likelihood(model(linspace))
        loc = predictions.mean.numpy()
        scale = predictions.stddev.numpy()
        lower, upper = priors.truncated_normal_uncertainty(0.0, 1.0, loc, scale)   
        ax.plot(linspace.detach().numpy(), loc, color=color, label=label)
        ax.fill_between(linspace.detach().numpy(), lower, upper, color=color, alpha=0.1)
    else:
        linspace = torch.linspace(50, 30000, 29950)
        with torch.no_grad(): loc = model(linspace)
        ax.plot(linspace.detach().numpy(), loc, color=color, label=label)
        
def plot_blank(ax):
    ax.imshow([[1]], cmap='gray', vmin=0, vmax=1)
    ax.set_axis_off()
        
def format_plot(ax, label):
    ax.set_xlim([50, 30000])
    ax.set_ylim([0.5, 1.0])
    ax.set_xscale('log')
    ax.set_xlabel('Number of training samples (log-scale)')
    ax.set_ylabel('{} AUROC'.format(label))
    
def print_coverage(model_objects, size, test_auroc):
    model, *likelihood_objects = model_objects
    label_map = { models.PowerLaw: 'Power law', models.Arctan: 'Arctan', models.GPPowerLaw: 'GP pow', models.GPArctan: 'GP arc' }
    label = label_map.get(type(model), 'Unknown') # Default label is 'Unknown' 
    if label.startswith('GP'):
        likelihood, = likelihood_objects
        with torch.no_grad(): predictions = likelihood(model(size*torch.ones(100)))
        loc = predictions.mean.numpy()
        scale = predictions.stddev.numpy()
        lower, upper = priors.truncated_normal_uncertainty(0.0, 1.0, loc, scale)   
        coverage_95 = metrics.coverage(test_auroc, lower, upper)
        lower, upper = priors.truncated_normal_uncertainty(0.0, 1.0, loc, scale, 0.1, 0.9)
        coverage_80 = metrics.coverage(test_auroc, lower, upper)
        print('{} 80% coverage at {}k: {:.2f}%'.format(label, size//1000, 100*coverage_80))
        print('{} 95% coverage at {}k: {:.2f}%'.format(label, size//1000, 100*coverage_95))
        
def grouped_mean_auroc(df):
    group_size = 3
    df['group'] = (df.index // group_size) + 1
    df = df.groupby('group').test_auroc.agg(lambda x: list(x)).reset_index()
    test_aurocs = np.array(df.test_auroc.tolist())
    # _, label, group
    mean_test_aurocs = np.mean(test_aurocs, axis=1)
    return mean_test_aurocs

def plot_min_max(ax, size, test_auroc):
    ax.plot([size, size], [np.min(test_auroc), np.max(test_auroc)], marker='_', color='black')