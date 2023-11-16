import argparse
import os
import ast
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# Importing our custom module(s)
import metrics
import folds
import logistic_regression
import utils

def finetune(attention, experiments_path, labels_path, n, random_state, lrs, max_slices):
    utils.makedir_if_not_exist(experiments_path)
    
    # Load labels.csv
    df = pd.read_csv(os.path.join(labels_path, 'labels.csv'), index_col='study_id')
    df.label = df.label.apply(lambda string: ast.literal_eval(string))
    
    # Train, validation, and test split
    if not 'Fold' in df.columns: df['Fold'] = folds.create_folds(df, random_state=random_state)
    train_df, val_df, test_df = folds.split_folds(df)
    
    # Subsample training data
    # TODO: Print warning if n > train_df.shape[0]
    if n < train_df.shape[0]: train_df = train_df.sample(n=n, random_state=random_state)
    
    # Load data
    train_dataset = utils.EncodedDataset(train_df, max_slices=max_slices)
    val_dataset = utils.EncodedDataset(val_df, max_slices=max_slices)
    test_dataset = utils.EncodedDataset(test_df, max_slices=max_slices)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Hyperparameters
    window_size = 30
    seeds = [1001, 2001, 3001, 4001, 5001]
    wds = np.append(np.logspace(0, -5, 6), 0)
    
    best_model_history_df = None
    best_val_loss = np.inf
    best_val_auroc = 0.0
    
    for seed, lr, wd in itertools.product(seeds, lrs, wds):

        torch.manual_seed(seed)

        train_loader = utils.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, collate_fn=utils.collate_fn)
        val_loader = utils.DataLoader(val_dataset, batch_size=len(val_dataset), collate_fn=utils.collate_fn)
        test_loader = utils.DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=utils.collate_fn)
        
        model = logistic_regression.LogisticRegression(attention)
        model.to(device)
        loss_func = nn.BCELoss()
        #loss_func = logistic_regression.BCEWithL1Loss(weight_decay=wd)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        columns = ['epoch', 'train_loss', 'train_BA', 'train_auroc', 'val_loss', 
                   'val_BA', 'val_auroc', 'test_loss', 'test_BA', 'test_auroc']
        model_history_df = pd.DataFrame(columns=columns)

        for epoch in range(3000):

            # Train
            train_loss = utils.train_one_epoch(model, device, optimizer, loss_func, train_loader)

            # Evaluate
            train_loss, train_labels, train_predictions = utils.evaluate(model, device, loss_func, train_loader)
            val_loss, val_labels, val_predictions = utils.evaluate(model, device, loss_func, val_loader)
            test_loss, test_labels, test_predictions = utils.evaluate(model, device, loss_func, test_loader)

            # Calculate balanced accuracies
            train_BA = metrics.get_balanced_accuracy(train_labels, train_predictions)
            thresholds, val_BA = metrics.get_balanced_accuracy(val_labels, val_predictions, return_thresholds=True)
            test_BA = metrics.get_balanced_accuracy(test_labels, test_predictions, thresholds=thresholds)

            # Calculate AUROCs
            train_auroc = metrics.get_auroc(train_labels, train_predictions)
            val_auroc = metrics.get_auroc(val_labels, val_predictions)
            test_auroc = metrics.get_auroc(test_labels, test_predictions)

            # Append evaluation metrics to DataFrame
            row = [epoch+1, train_loss, train_BA, train_auroc, val_loss, val_BA, val_auroc, test_loss, test_BA, test_auroc]
            model_history_df.loc[epoch] = row
            print(model_history_df.iloc[epoch])
            
            # Stopping criterion
            if model_history_df.shape[0] > window_size:
                train_loss_list = model_history_df.tail(4).train_loss.to_list()
                if np.all([abs(train_loss_list[i] - train_loss_list[i-1]) < 1e-6 for i in range(1, 4)]):
                    break

        val_loss = model_history_df.val_loss.values
        averaged_loss = [np.mean(val_loss[index-window_size:index]) for index in range(window_size, len(val_loss))]
        val_auroc = model_history_df.val_auroc.values
        averaged_auroc = [np.mean(val_auroc[index-window_size:index]) for index in range(window_size, len(val_auroc))]
        print(model_history_df.iloc[window_size+np.argmax(averaged_auroc)])

        if val_auroc[window_size+np.argmax(averaged_auroc)] > best_val_auroc:
            best_model_history_df = model_history_df
            best_val_auroc = val_auroc[window_size+np.argmax(averaged_auroc)]
            hyperparameters = {'seed': seed, 'lr': lr, 'wd': wd}
            print(hyperparameters, file=open('{}/n={}_random_state={}.txt'.format(experiments_path, n, random_state), 'w'), flush=True)
            best_model_history_df.to_csv('{}/n={}_random_state={}.csv'.format(experiments_path, n, random_state))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='finetune_3D.py')
    parser.add_argument('--attention', action='store_true', default=False, help='Whether or not to use attention pooling (default: False)')
    parser.add_argument('--experiments_path', required=True, type=str)
    parser.add_argument('--labels_path', required=True, type=str)
    parser.add_argument('--lrs', default=[0.05, 0.01], help='Learning rates (default: [0.05, 0.01])', nargs='+', type=float)
    parser.add_argument('--max_slices', default=50, help='Maximum number of slices (default: 50)', type=int)
    parser.add_argument('--n', help='Number of training samples', required=True, type=int)
    parser.add_argument('--random_state', help='Random state', required=True, type=int)
    
    # Print model hyperparameters to file
    args = parser.parse_args()
    print(args)

    finetune(args.attention, args.experiments_path, args.labels_path, args.n, args.random_state, args.lrs, args.max_slices)