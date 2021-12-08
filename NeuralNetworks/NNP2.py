from functools import lru_cache
from typing import DefaultDict
from pandas.core.frame import DataFrame
from scipy.special import expit as sigmoid
import numpy as np
import pandas as pd
from collections import defaultdict
class ThreeNeuralNetwork:
    def __init__(self, h_size, i_wgt, n_features) -> None:
        self.wghts = defaultdict(dict)
        self.h_size = h_size
        for j in range(0, h_size):
            self.wghts[1][j] = np.full(1 + n_features, i_wgt)
            self.wghts[2][j] = np.full(h_size, i_wgt)
        self.wghts[3][0] = np.full(h_size, i_wgt)

    def predict(self, row : DataFrame):
        self.vec = row[:-1].to_numpy()
        self.vec.insert(1, 0)
        self.__get_neuron_value__.cache_clear()
        self.__get_neuron_value__.cache_clear()

    @lru_cache(None)
    def __get_neuron_value__(self, layer, i):
        if not i and layer != 3:
            return 1
        if layer == 0:
            return self.vec[i]
        wgt = self.wghts[layer][i]
        prev = self.__get_neuron_vector__(layer-1)
        return sigmoid(np.dot(wgt, prev))
        
    @lru_cache(None)
    def __get_neuron_vector__(self, layer):
        if layer == 0:
            return self.vec
        return np.array([self.__get_neuron_value__(layer, i) for i in range(0, self.h_size)])
        
        
        
NN = ThreeNeuralNetwork(3, 1, 2)
NN.vec = np.array([-.6, .5, .1])
NN.wghts[1][1] = np.array([-1, -2, -3])
NN.wghts[1][2] = np.array([1, 2, 3])
NN.wghts[2][1] = np.array([-1, -2, -3])
NN.wghts[2][2] = np.array([1, 2, 3])
NN.wghts[3, 0] = np.array([-1, 2, -1.5])
print(NN.__get_neuron_value__(3, 0))
        

