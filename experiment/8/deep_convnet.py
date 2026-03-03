import sys,os
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *

class DeepConvNet:
    def __init__(self,input_dim=(1,28,28),
                 conv_param_1={"filter_num":16,"filter_size":3,"pad":1,"stride":1},
                 conv_param_2={"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
                 conv_param_3={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
                 conv_param_4={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
                 conv_param_5={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
                 conv_param_6={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
                 hidden_size=50,output_size=10):
        pre_node_nums=np.array([1*3*3,
                                16*3*3,16*3*3,
                                32*3*3,32*3*3,
                                64*3*3,64*4*4,
                                hidden_size])
        wight_init_scales=np.sqrt(2.0/pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4,
                                          conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = wight_init_scales[idx] * \
                                            np.random.randn(conv_param['filter_num'],
                                                            pre_channel_num, conv_param['filter_size'],
                                                            conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

