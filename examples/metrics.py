import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(output, target):
    pred = np.argmax(output.data, axis=1)
    acc = accuracy_score(target, pred)  # , labels=np.unique(pred))
    return acc


def precision(output, target, average='micro'):
    pred = np.argmax(output.data, axis=1)
    # one_hot = np.zeros(output.size())
    # one_hot[np.arange(output.size(0)), target] = 1
    p = precision_score(target, pred, labels=range(output.data.shape[1]), average=average)  # , labels=np.unique(pred))
    return p


def recall(output, target, average='micro'):
    pred = np.argmax(output.data, axis=1)
    r = recall_score(target, pred, labels=range(output.data.shape[1]), average=average)  # , labels=np.unique(pred))
    return r


def f1(output, target, average='micro'):
    pred = np.argmax(output.data, axis=1)
    f = f1_score(target, pred, labels=range(output.data.shape[1]), average=average)  # , labels=np.unique(pred))
    return f
