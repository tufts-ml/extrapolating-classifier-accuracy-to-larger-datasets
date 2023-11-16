import sys
sys.path.append('../src/')
import os
import math
import numpy as np
import pandas as pd

from models import *
from priors import *
from utils import *

if __name__=='__main__':
    
    experiments_path = '/cluster/home/eharve06/extrapolating-classifier-accuracy-to-bigger-datasets/experiments'
    models_path = os.path.join(os.path.dirname(experiments_path), 'models')
    datasets = [('ChestX-ray14', 'ChestX-ray14_short_range.csv'),
                ('Chest_X-Ray', 'Chest_X-Ray_short_range.csv'),
                ('BUSI', 'BUSI_short_range.csv'),
                ('TMED-2', 'TMED-2_short_range.csv'),
                ('OASIS-3', 'OASIS-3_short_range.csv'),
                ('Pilot', 'Pilot_short_range.csv')]
    datasets = [('OASIS-3', 'OASIS-3_short_range.csv'),
                ('Pilot', 'Pilot_short_range.csv')]
    models = [('PowerLaw', train_PowerLaw), 
              ('Arctan', train_Arctan), 
              ('GPPowerLaw', train_GPPowerLaw), 
              ('GPArctan', train_GPArctan)]
    labels = [['Atelectasis', 'Effusion', 'Infiltration'],
              ['Bacterial', 'Viral'],
              ['Normal', 'Benign', 'Malignant'],
              ['PLAX', 'PSAX', 'A4C', 'A2C'],
              ['Alzheimer’s'],
              ['WMD', 'CBI']]
    labels = [['Alzheimer’s'],
              ['WMD', 'CBI']]
    
    for datasets_index, (dataset_name, filename) in enumerate(datasets):
        df = load_experiment(os.path.join(experiments_path, filename))
        # Take mean of each random seed at each dataset size
        df = df.groupby('n').agg(lambda x: list(x))
        df.test_auroc = df.test_auroc.apply(lambda x: np.mean(x, axis=0))
        df.random_state = df.random_state.apply(lambda x: 'mean')
        df = df.reset_index()
        for label_index, label_name in enumerate(labels[datasets_index]):
            # Split data
            X_train, y_train, X_test, y_test = split_df(df, index=label_index)
            for model_name, training_func in models:
                print(model_name)
                func_results = training_func(X_train, y_train)
                if len(func_results) == 2: model, losses = func_results
                else: likelihood, model, losses = func_results
                model_filename = '{}_{}_{}.pt'.format(dataset_name, label_name, model_name)
                torch.save(model.state_dict(), os.path.join(models_path, model_filename))