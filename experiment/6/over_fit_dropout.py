import sys,os


sys.path.append(os.pardir)
import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True)
x_train=x_train[:300]
t_train=t_train[:300]

use_dropout=True
dropout_ratio=0.2

network=MultiLayerNetExtend
