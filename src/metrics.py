import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def rmse(labels, predictions):
    assert labels.shape == predictions.shape,\
    'labels.shape != predictions.shape'
    squared_diff = np.square(labels - predictions)
    mse = np.mean(squared_diff, axis=0)
    rmse = np.sqrt(mse)
    return rmse

def coverage(labels, lower, upper):
    return len(labels[(labels>=lower)&(labels<=upper)])/len(labels)

def get_balanced_accuracy(labels, predictions, thresholds=None, return_thresholds=False):
    labels = np.array(labels)
    predictions = np.array(predictions)
    assert labels.shape == predictions.shape,\
    'labels.shape != predictions.shape'
    
    # Calculate threshold that maximizes tpr - fpr for each label if no thresholds
    if thresholds is None:
        _, num_labels = labels.shape
        roc_curves = [roc_curve(labels[:,label_index], predictions[:,label_index])\
                  for label_index in range(num_labels)]
        thresholds = np.array([thresholds[np.argmax(tpr - fpr)]\
                               for fpr, tpr, thresholds in roc_curves])
        
    predictions = np.where(predictions >= thresholds, 1, 0)
    TP = np.sum(np.where(labels == 1, predictions, 0), axis=0)
    TN = np.sum(np.where(labels == 0, 1-predictions, 0), axis=0)
    FP = np.sum(np.where(labels == 0, predictions, 0), axis=0)
    FN = np.sum(np.where(labels == 1, 1-predictions, 0), axis=0)
    sensitivity = np.where(TP+FN != 0, TP/(TP+FN), 0)
    specificity = np.where(TN+FP != 0, TN/(TN+FP), 0)

    return (0.5 * (sensitivity + specificity)) if return_thresholds is False else (thresholds, 0.5 * (sensitivity + specificity))

def get_auroc(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)
    assert labels.shape == predictions.shape,\
    'labels.shape != predictions.shape'
    
    _, num_labels = labels.shape
    aurocs = np.array([roc_auc_score(labels[:,label_index], predictions[:,label_index])\
                       for label_index in range(num_labels)])
    return aurocs