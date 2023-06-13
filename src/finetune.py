import argparse
import os
import numpy as np
import pandas as pd

import ast
import itertools
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from evaluation_metrics import *
from folds import *
from utils import *

def finetune(directory, n, random_state):
    # Load labels.csv
    df = pd.read_csv(os.path.join(directory, 'labels.csv'), index_col='study_id')
    df.label = df.label.apply(lambda string: ast.literal_eval(string))
    
    # Train, validation, and test split
    df['Fold'] = create_folds(df, random_state)
    train_df, val_df, test_df = split_folds(df)
    
    # Subsample training data
    assert n <= train_df.shape[0], 'n={} is greater than number of training samples'.format(n)
    train_df = train_df.sample(n=n, random_state=random_state)
        
    # Load data
    X_train, y_train = load_dataset(train_df)
    X_val, y_val = load_dataset(val_df)
    X_test, y_test = load_dataset(test_df)

    # Hyperparameters
    states = [1001, 2001, 3001, 4001, 5001]
    Cs = np.logspace(5, -5, 11)
    max_iters = np.logspace(1, 3.69897000434, 10, dtype=int)

    best_c = 0
    best_clf = None
    best_clf_performance = 0.0

    for state, C, max_iter in itertools.product(states, Cs, max_iters):
        clf = OneVsRestClassifier(LogisticRegression(penalty='l2', C=C, random_state=state, solver='lbfgs', max_iter=max_iter))
        clf.fit(X_train, y_train)

        train_predictions = clf.predict_proba(X_train)
        val_predictions = clf.predict_proba(X_val)
        
        if y_test.shape[-1] == 1 and train_predictions.shape[-1] == 2:
            train_predictions = train_predictions[:,1][:,np.newaxis]
            val_predictions = val_predictions[:,1][:,np.newaxis]
        
        # Calculate balanced accuracies
        train_BA = get_balanced_accuracy(y_train, train_predictions)
        val_BA = get_balanced_accuracy(y_val, val_predictions)

        # Calculate AUROCs
        train_auroc = get_auroc(y_train, train_predictions)
        val_auroc = get_auroc(y_val, val_predictions)
        
        # Save best model
        if (np.sum(val_BA)+np.sum(val_auroc))/(len(val_BA)+len(val_auroc)) > best_clf_performance:
            best_c = C
            best_clf = clf
            best_clf_performance = (np.sum(val_BA)+np.sum(val_auroc))/(len(val_BA)+len(val_auroc))
    
    print(best_c)
   
    train_predictions = best_clf.predict_proba(X_train)
    val_predictions = best_clf.predict_proba(X_val)
    test_predictions = best_clf.predict_proba(X_test)
    
    if y_test.shape[-1] == 1 and train_predictions.shape[-1] == 2:
        train_predictions = train_predictions[:,1][:,np.newaxis]
        val_predictions = val_predictions[:,1][:,np.newaxis]
        test_predictions = test_predictions[:,1][:,np.newaxis]

    # Calculate balanced accuracies
    train_BA = get_balanced_accuracy(y_train, train_predictions)
    thresholds, val_BA = get_balanced_accuracy(y_val, val_predictions, return_thresholds=True)
    test_BA = get_balanced_accuracy(y_test, test_predictions, thresholds=thresholds)
    
    # Calculate AUROCs
    train_auroc = get_auroc(y_train, train_predictions)
    val_auroc = get_auroc(y_val, val_predictions)
    test_auroc = get_auroc(y_test, test_predictions)
    
    return train_BA, train_auroc, val_BA, val_auroc, test_BA, test_auroc