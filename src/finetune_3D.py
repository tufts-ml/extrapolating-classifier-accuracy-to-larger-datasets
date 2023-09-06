import argparse
import os
import numpy as np
import pandas as pd

import ast
import itertools
# Importing our custom module(s)
import metrics
import folds
import logistic_regression
import utils

def finetune(experiments_path, labels_path, lr, n, random_state):
    # Load labels.csv
    df = pd.read_csv(os.path.join(labels_path, 'labels.csv'), index_col='study_id')
    df.label = df.label.apply(lambda string: ast.literal_eval(string))
    
    # Train, validation, and test split
    df['Fold'] = folds.create_folds(df, random_state=random_state)
    train_df, val_df, test_df = folds.split_folds(df)
    
    # Subsample training data
    # TODO: Print warning if n > train_df.shape[0]
    if n < train_df.shape[0]: train_df = train_df.sample(n=n, random_state=random_state)
    
    # Load data
    train_dataset = utils.EncodedDataset(train_df)
    val_dataset = utils.EncodedDataset(val_df)
    test_dataset = utils.EncodedDataset(test_df)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Hyperparameters
    seeds = [1001]
    wd1s = np.append(np.logspace(0, -5, 6), 0)

    best_model_history_df = None
    best_val_performance = np.inf
    
    for seed, wd1 in itertools.product(seeds, wd1s):
        #print('seed: {}, wd1: {}, wd2: {}'.format(seed, wd1, wd2))
        torch.manual_seed(seed)

        train_loader = utils.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = utils.DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)
        test_loader = utils.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        
        model = logistic_regression.LogisticRegression()
        model.to(device)
        loss_func = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd1)

        columns = ['epoch', 'train_loss', 'train_BA', 'train_auroc', 'val_loss', 
                   'val_BA', 'val_auroc', 'test_loss', 'test_BA', 'test_auroc']
        model_history_df = pd.DataFrame(columns=columns)

        for epoch in range(1000):

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
            #print(model_history_df.iloc[epoch])
            
            # Stopping criterion
            if model_history_df.shape[0] > 30:
                train_loss_list = model_history_df.tail(4).train_loss.to_list()
                if np.all([abs(train_loss_list[i] - train_loss_list[i-1]) < 1e-6 for i in range(1, 4)]):
                    break

        val_performance = np.array(model_history_df.val_loss.to_list())
        averaged_performance = np.array([sum(val_performance[index-30:index]) for index in range(30, len(val_performance))])
        #print(model_history_df.iloc[30+np.argmin(averaged_performance)])

        if val_performance[30+np.argmin(averaged_performance)] < best_val_performance:
            best_model_history_df = model_history_df
            best_val_performance = val_performance[30+np.argmin(averaged_performance)]
            hyperparameters = {'seed': seed, 'lr': lr, 'wd1': wd1}
            print(hyperparameters, file=open('{}/n={}_random_state={}.txt'.format(experiments_path, n, random_state), 'w'), flush=True)
            best_model_history_df.to_csv('{}/n={}_random_state={}.csv'.format(experiments_path, n, random_state))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='finetune_3D.py')
    parser.add_argument('--experiments_path', required=True, type=str)
    parser.add_argument('--labels_path', required=True, type=str)
    parser.add_argument('--lr', default=0.1, help='learning rate', type=float)
    parser.add_argument('--n', help='number of epochs', required=True, type=int)
    parser.add_argument('--random_state', help='random state', required=True, type=int)
    
    # Print model hyperparameters to file
    args = parser.parse_args()
    print(args)

    finetune(args.experiments_path, args.labels_path, args.lr, args.n, args.random_state)