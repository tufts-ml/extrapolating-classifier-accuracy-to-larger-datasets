import argparse
import os
import numpy as np
import pandas as pd

import ast
import itertools

from evaluation_metrics import *
from folds import *
from logistic_regression import *
from utils import *

def finetune(directory, n, random_state):
    # Load labels.csv
    df = pd.read_csv(os.path.join(directory, 'labels.csv'), index_col='study_id')
    df.label = df.label.apply(lambda string: ast.literal_eval(string))
    
    # Train, validation, and test split
    df['Fold'] = create_folds(df, random_state=random_state)
    train_df, val_df, test_df = split_folds(df)
    
    # Subsample training data
    assert n <= train_df.shape[0], 'n={} is greater than {} training samples'.format(n, train_df.shape[0])
    train_df = train_df.sample(n=n, random_state=random_state)
    
    # Load data
    train_dataset = EncodedDataset(train_df)
    val_dataset = EncodedDataset(val_df)
    test_dataset = EncodedDataset(test_df)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Hyperparameters
    seeds = [1001, 2001, 3001, 4001, 5001]
    wd1s = np.append(np.logspace(0, -5, 6), 0)
    wd2s = np.append(np.logspace(0, -5, 6), 0)

    best_model_history_df = None
    best_val_performance = 0.0
    
    for seed, wd1, wd2 in itertools.product(seeds, wd1s, wd2s):
        print('seed: {}, wd1: {}, wd2: {}'.format(seed, wd1, wd2))

        torch.manual_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        
        model = LogisticRegression(attention=True, num_classes=1)
        model.to(device)
        loss_func = nn.BCELoss()
        optimizer = torch.optim.SGD([
            {'params': model.fc.parameters(), 'weight_decay': wd1},
            {'params': model.attention_pooling[0].parameters(), 'weight_decay': wd2},
        ], lr=0.1, momentum=0.9)

        columns = ['epoch', 'train_loss', 'train_BA', 'train_auroc', 'val_loss', 
                   'val_BA', 'val_auroc', 'test_loss', 'test_BA', 'test_auroc']
        model_history_df = pd.DataFrame(columns=columns)

        for epoch in range(1000):

            # Train
            train_loss = train_one_epoch(model, device, optimizer, loss_func, train_loader)

            # Evaluate
            train_loss, train_labels, train_predictions = evaluate(model, device, loss_func, train_loader)
            val_loss, val_labels, val_predictions = evaluate(model, device, loss_func, val_loader)
            test_loss, test_labels, test_predictions = evaluate(model, device, loss_func, test_loader)

            # Calculate balanced accuracies
            train_BA = get_balanced_accuracy(train_labels, train_predictions)
            thresholds, val_BA = get_balanced_accuracy(val_labels, val_predictions, return_thresholds=True)
            test_BA = get_balanced_accuracy(test_labels, test_predictions, thresholds=thresholds)

            # Calculate AUROCs
            train_auroc = get_auroc(train_labels, train_predictions)
            val_auroc = get_auroc(val_labels, val_predictions)
            test_auroc = get_auroc(test_labels, test_predictions)

            # Append evaluation metrics to DataFrame
            row = [epoch+1, train_loss, train_BA, train_auroc, val_loss, val_BA, val_auroc, test_loss, test_BA, test_auroc]
            model_history_df.loc[epoch] = row
            print(model_history_df.iloc[epoch])

            # Stopping criterion
            if model_history_df.shape[0] > 3:
                train_loss_list = model_history_df.tail(4).train_loss.to_list()
                if np.all([abs(train_loss_list[i] - train_loss_list[i-1]) < 1e-6 for i in range(1, 4)]):
                    break

        val_performance = np.sum(np.array(model_history_df.val_auroc.to_list()), axis=-1)
        averaged_performance = np.array([sum(val_performance[index-30:index]) for index in range(30, len(val_performance))])
        print(model_history_df.iloc[30+np.argmax(averaged_performance)])

        if val_performance[30+np.argmax(averaged_performance)] > best_val_performance:
            best_model_history_df = model_history_df
            best_val_performance = val_performance[30+np.argmax(averaged_performance)]
            hyperparameters = {'wd1': wd1,
                               'wd2': wd2}
            print(hyperparameters, file=open('/cluster/home/eharve06/extrapolating-classifier-accuracy-to-bigger-datasets/experiments/OASIS-3/n={}_random_state={}.txt'.format(n, random_state), 'w'))
            best_model_history_df.to_csv('/cluster/home/eharve06/extrapolating-classifier-accuracy-to-bigger-datasets/experiments/OASIS-3/n={}_random_state={}.csv'.format(n, random_state))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='finetune_3D.py')
    parser.add_argument('--directory', default='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3', type=str)
    parser.add_argument('--n', help='number of epochs', required=True, type=int)
    parser.add_argument('--random_state', help='random state', required=True, type=int)
    
    # Print model hyperparameters to file
    args = parser.parse_args()
    print(args)

    finetune(args.directory, args.n, args.random_state)