import numpy as np
def mean_squared(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entroy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))