import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

def load_experiment(path):
    df = pd.read_csv(path, index_col='Unnamed: 0')
    df.train_BA = df.train_BA.apply(lambda string: np.fromstring(string[1:-1], sep=' '))
    df.train_auroc = df.train_auroc.apply(lambda string: np.fromstring(string[1:-1], sep=' '))
    df.val_BA = df.val_BA.apply(lambda string: np.fromstring(string[1:-1], sep=' '))
    df.val_auroc = df.val_auroc.apply(lambda string: np.fromstring(string[1:-1], sep=' '))
    df.test_BA = df.test_BA.apply(lambda string: np.fromstring(string[1:-1], sep=' '))
    df.test_auroc = df.test_auroc.apply(lambda string: np.fromstring(string[1:-1], sep=' '))
    return df

def rmse(labels, predictions):
    assert labels.shape == predictions.shape,\
    'labels.shape != predictions.shape'
    squared_diff = np.square(labels - predictions)
    mse = np.mean(squared_diff, axis=0)
    rmse = np.sqrt(mse)
    return rmse

def calc_coverage(labels, lower, upper):
    return len(labels[(labels>=lower)&(labels<=upper)])/len(labels)

def load_dataset(df):
    X = torch.vstack([torch.load(path).float() for path in df.path.to_list()]).detach().numpy()
    y = torch.tensor(df.label.to_list()).detach().numpy()
    return X, y

class EncodedDataset(Dataset):
    def __init__(self, df):
        self.path = df.path.to_list()
        self.label = df.label.to_list()
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        image = torch.load(self.path[index], map_location='cpu').float().detach().numpy()
        slices = min(111, image.shape[0])
        linspace = np.linspace(start=0, stop=image.shape[0]-1, num=slices, dtype=int)
        return image[linspace], self.label[index]

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

            if device.type == 'cuda':
                loss = loss.cpu()
                labels = labels.cpu()
                outputs = outputs.cpu()

            for output, label in zip(outputs.cpu(), labels):
                label_list.append(label.numpy().astype(int))
                prediction_list.append(output.numpy())

            running_loss += loss.numpy()
            
    return running_loss, label_list, prediction_list