import argparse
import os
import numpy as np
import pandas as pd

import ast # Package that converts a string in list format into a list
import itertools
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
# Importing our custom module(s)
import metrics
import folds
import utils

def finetune(labels_path, n, random_state):
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
    X_train, y_train = utils.load_dataset(train_df)
    X_val, y_val = utils.load_dataset(val_df)
    X_test, y_test = utils.load_dataset(test_df)

    # Hyperparameters
    states = [1001, 2001, 3001, 4001, 5001]
    Cs = np.logspace(5, -5, 11)
    max_iters = np.logspace(1, 3.69897000434, 10, dtype=int)
    
    _, num_labels = y_train.shape
    best_c = [0]*num_labels
    best_iter = [0]*num_labels
    best_clf = [None]*num_labels
    best_clf_performance = [0.0]*num_labels

    for state, C, max_iter in itertools.product(states, Cs, max_iters):
        clf = OneVsRestClassifier(LogisticRegression(penalty='l2', C=C, class_weight='balanced', random_state=state, solver='lbfgs', max_iter=max_iter))
        clf.fit(X_train, y_train)

        train_predictions = clf.predict_proba(X_train)
        val_predictions = clf.predict_proba(X_val)
        
        # TODO: Filter the predictions to only include the positive class (1) and exclude the negative class (0) from sklearn results (there are no 2D datasets in our experiments with one label)
        
        # Calculate balanced accuracies
        train_BA = metrics.get_balanced_accuracy(y_train, train_predictions)
        val_BA = metrics.get_balanced_accuracy(y_val, val_predictions)

        # Calculate AUROCs
        train_auroc = metrics.get_auroc(y_train, train_predictions)
        val_auroc = metrics.get_auroc(y_val, val_predictions)
        
        # Save best model for each label
        for label_index in range(num_labels):
            if val_auroc[label_index] > best_clf_performance[label_index]:
                best_c[label_index] = C
                best_iter[label_index] = max_iter
                best_clf[label_index] = clf
                best_clf_performance[label_index] = val_auroc[label_index]
    
    train_predictions = [best_clf[label_index].predict_proba(X_train)[:,label_index] for label_index in range(num_labels)]
    val_predictions = [best_clf[label_index].predict_proba(X_val)[:,label_index] for label_index in range(num_labels)]
    test_predictions = [best_clf[label_index].predict_proba(X_test)[:,label_index] for label_index in range(num_labels)]
    train_predictions = np.transpose(train_predictions)
    val_predictions = np.transpose(val_predictions)
    test_predictions = np.transpose(test_predictions)

    # Calculate balanced accuracies
    train_BA = metrics.get_balanced_accuracy(y_train, train_predictions)
    thresholds, val_BA = metrics.get_balanced_accuracy(y_val, val_predictions, return_thresholds=True)
    test_BA = metrics.get_balanced_accuracy(y_test, test_predictions, thresholds=thresholds)
    
    # Calculate AUROCs
    train_auroc = metrics.get_auroc(y_train, train_predictions)
    val_auroc = metrics.get_auroc(y_val, val_predictions)
    test_auroc = metrics.get_auroc(y_test, test_predictions)
    
    return train_BA, train_auroc, val_BA, val_auroc, test_BA, test_auroc