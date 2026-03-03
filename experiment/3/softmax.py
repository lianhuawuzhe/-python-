import numpy as np
def softmax(a):
    c=np.max(a)
    return np.exp(a-c)/np.sum(np.exp(a-c))