import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
node_num=100
x=np.random.randn(1000,100)/np.sqrt(node_num)

hidden_layer_size=5
activations={}

for i in range(hidden_layer_size):
    if i !=0:
        x=activations[i-1]
    w=np.random.randn(node_num,node_num)*1
    z=np.dot(x,w)
    a=sigmoid(z)
    activations[i]=a

for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    if i !=0:plt.yticks([],[])
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()