import numpy as np

def derive_class_weight(y_train):
    cw = {i: 1 - (np.sum(y_train[..., i]) / len(y_train)) for i in range(len(y_train[0]))}
    return cw