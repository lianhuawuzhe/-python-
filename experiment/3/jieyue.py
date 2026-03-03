# def step_function(x):
#     if x>0:
#         return 1
#     else:
#         return 0

import matplotlib.pylab as plt
import numpy as np
# def step_function(x):
#     y=x>0
#     return y.astype(np.int)
# # 这里np.int被弃置了吧，应该用int

def step_function(x):
    return np.array(x>0,dtype=int)
x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.2,1.2)
plt.show()