import numpy as np
import pandas as pd

def create_folds(df, random_state=42):

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')
        
    rows, columns = df.shape
    
    numpy_labels = np.array([np.array(label) for label in df.label.to_list()])
    unique_labels, counts = np.unique(numpy_labels, axis=0, return_counts=True)
    folds_for_each_label = [random_state.choice(10, count) for count in counts]
    
    folds = -1*np.ones(rows)

    for unique_label_index, unique_label in enumerate(unique_labels):
        folds[np.all(numpy_labels == unique_label, axis=1)] = folds_for_each_label[unique_label_index]
        
    return folds.astype(int)

def split_folds(df, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], val_indices=[8], test_indices=[9]):
    
    train_df = df[df.Fold.isin(train_indices)].copy()
    val_df = df[df.Fold.isin(val_indices)].copy()
    test_df = df[df.Fold.isin(test_indices)].copy()
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df