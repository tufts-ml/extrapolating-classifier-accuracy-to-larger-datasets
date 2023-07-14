import numpy as np
import pandas as pd

def create_folds(df, index_name='subject_id', random_state=42):

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')
        
    # Create DataFrame with the most frequent label for each unique subject_id
    df.index.name = index_name
    grouped = df.groupby(index_name)['label'].agg(lambda x: x.value_counts().idxmax()).reset_index()

    rows, columns = grouped.shape

    numpy_labels = np.array([np.array(label) for label in grouped.label.to_list()])
    unique_labels, counts = np.unique(numpy_labels, axis=0, return_counts=True)
    folds_for_each_label = [random_state.choice(np.tile(np.arange(10), count//10+1)[:count], count, replace=False) for count in counts]

    fold_numbers = -1*np.ones(rows)

    for unique_label_index, unique_label in enumerate(unique_labels):
            fold_numbers[np.all(numpy_labels == unique_label, axis=1)] = folds_for_each_label[unique_label_index]

    return df.index.map(dict(zip(grouped[index_name].to_numpy(), fold_numbers))).astype(int)

def split_folds(df, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], val_indices=[8], test_indices=[9]):
    
    assert 'Fold' in df.columns, 'DataFrame is missing Fold column'
    
    train_df = df[df.Fold.isin(train_indices)].copy()
    val_df = df[df.Fold.isin(val_indices)].copy()
    test_df = df[df.Fold.isin(test_indices)].copy()
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df