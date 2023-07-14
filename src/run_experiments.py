import argparse
import itertools
import pandas as pd
from finetune_2D import *
directory = '/cluster/tufts/hugheslab/eharve06/encoded_images/'
columns = ['n', 'random_state', 'train_BA', 'train_auroc', 'val_BA', 'val_auroc', 'test_BA', 'test_auroc']
df = pd.DataFrame(columns=columns)
ns = [200, 240, 280, 320, 360, 800, 1600, 3200, 6400, 12800, 25600]
random_states = [1001, 2001, 3001]
for model_index, (n, random_state) in enumerate(itertools.product(ns, random_states)):
    train_BA, train_auroc, val_BA, val_auroc, test_BA, test_auroc = finetune(directory, n, random_state)
    row = [n, random_state, train_BA, train_auroc, val_BA, val_auroc, test_BA, test_auroc]
    df.loc[model_index] = row
    print(df.loc[model_index])
df.to_csv('/cluster/home/eharve06/extrapolating-classifier-accuracy-to-bigger-datasets/experiments/ChestX-ray14_long_range.csv')