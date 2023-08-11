import argparse
import itertools
import pandas as pd
from finetune_2D import *
# sbatch < run_experiments.sh
directory = '/cluster/tufts/hugheslab/eharve06/encoded_images/'
columns = ['n', 'random_state', 'train_BA', 'train_auroc', 'val_BA', 'val_auroc', 'test_BA', 'test_auroc']
df = pd.DataFrame(columns=columns)
ns = [60, 94, 147, 230, 360, 804, 1796, 4010, 8955, 20000]
random_states = [1001, 2001, 3001]
i = 0
#ns = [20000]
#random_states = [1001, 2001, 3001, 4001, 5001, 6001, 7001, 8001, 9001, 10001, 11001, 12001, 13001, 14001, 15001, 16001, 17001, 18001, 19001, 20001, 21001, 22001, 23001, 24001, 25001, 26001, 27001, 28001, 29001, 30001]
#i = 9
for model_index, (n, random_state) in enumerate(itertools.product(ns, random_states)):
    train_BA, train_auroc, val_BA, val_auroc, test_BA, test_auroc = finetune(directory, n, random_state+(i*30001))
    row = [n, random_state+(i*30001), train_BA, train_auroc, val_BA, val_auroc, test_BA, test_auroc]
    df.loc[model_index] = row
    print(df.loc[model_index])
    df.to_csv('/cluster/home/eharve06/extrapolating-classifier-accuracy-to-bigger-datasets/experiments/ChestX-ray14_long_range.csv')
    #df.to_csv('/cluster/home/eharve06/extrapolating-classifier-accuracy-to-bigger-datasets/experiments/ChestX-ray14_20k_i={}.csv'.format(i))