import numpy as np
X=np.array([1.0,0.5])
w1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
A1=np.dot(X,w1)+B1
def sigmod(x):
    return 1/(1+np.exp(-x))
Z1=sigmod(A1)
print(A1)
print(Z1)


w2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
A2=np.dot(Z1,w2)+B2
Z2=sigmod(A2)


def identity_function(x):
    return x
w3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])
A3=np.dot(Z2,w3)+B3
Y=identity_function(A3)
print(Y)