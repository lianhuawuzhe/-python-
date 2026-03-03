import matplotlib.pylab as plt
import numpy as np
def sigmod(x):
    return 1/(1+np.exp(-x))
x=np.arange(-5.0,5.0,0.1)
y=sigmod(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()