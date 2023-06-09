import numpy as np
import pandas as pd
import torch

def load_dataset(df):
    X = torch.vstack([torch.load(path).float() for path in df.path.to_list()]).detach().numpy()
    y = torch.tensor(df.label.to_list()).detach().numpy()
    return X, y