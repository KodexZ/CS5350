from functools import lru_cache
from pandas.core.frame import DataFrame
from scipy.special import expit as sigmoid
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.random import normal
class ThreeNeuralNetwork:
    def __init__(self, h_size, n_features, train : DataFrame, lrate = .01, epoch = 10, d=4) -> None:
        self.wghts = defaultdict(dict)
        self.h_size = h_size
        self.n_features = n_features
        self.olrate = lrate
        for j in range(1, h_size):
            self.wghts[1][j] = np.full(1 + n_features, 0)
            self.wghts[2][j] = np.full(h_size, 0)
        self.wghts[3][1] = np.full(h_size, 0)

        
        for i in range(0, epoch):
            here = train.sample(frac=1)
            for j in range(0, len(train)):
                row = here.iloc[j]
                pred = self.predict(row[:-1])
                self.__get_partial_neuron__.cache_clear()
                self.__get_partial_weight__.cache_clear()
                real = row[-1]
                partials = self.__get_all_partials__(pred, real)
                #print(pred - real)
                curlrate = lrate/(1+(i*lrate/d))
                for i in range(1, 4):
                    for j in range(1, len(self.wghts[i]) + 1):
                        self.wghts[i][j] = np.add(self.wghts[i][j], -curlrate*np.array(partials[i-1][j-1]))
            #print(self.wghts)
        
        
        
    def predict(self, row : DataFrame):
        #print(f'last row is {row}')
        self.vec = row.to_numpy()
        self.vec = np.insert(self.vec, 0, 1)
        self.__get_neuron_value__.cache_clear()
        self.__get_neuron_vector__.cache_clear()
        return self.__get_neuron_value__(3, 1)

    @lru_cache(None)
    def __get_neuron_value__(self, layer, i):
        if not i and layer != 3:
            return 1
        if layer == 0:
            return self.vec[i]
        wgt = self.wghts[layer][i]
        prev = self.__get_neuron_vector__(layer-1)
        if layer == 3:
            return np.dot(wgt, prev)
        #print(wgt)
        #print(prev)
        #print(self.vec)
        #print(layer)
        #print(i)
        return sigmoid(np.dot(wgt, prev))
        
    @lru_cache(None)
    def __get_neuron_vector__(self, layer):
        if layer == 0:
            return self.vec
        return np.array([self.__get_neuron_value__(layer, i) for i in range(0, self.h_size)])
    
    @lru_cache(None)
    def __get_partial_weight__(self, layer, to_neuron, weight_i, y, ystar):
        if layer == 3:
            return self.__get_neuron_value__(layer - 1, weight_i)*(y-ystar)
        else:
            wgt = self.wghts[layer][to_neuron]
            prev = self.__get_neuron_vector__(layer-1)
            s = np.dot(wgt, prev)
            return sigmoid(s)*(1-sigmoid(s))*prev[weight_i]*self.__get_partial_neuron__(layer, to_neuron, y, ystar)


    @lru_cache(None)
    def __get_partial_neuron__(self, layer, neuron, y, ystar):
        if layer == 2:
            return (y-ystar)*self.wghts[3][1][neuron]
        else:
            arr = []
            for i in range(1, self.h_size):
                wgt = self.wghts[layer+1][i]
                prev = self.__get_neuron_vector__(layer)
                s = np.dot(wgt, prev)
                ta = self.__get_partial_neuron__(layer+1, i, y, ystar)*sigmoid(s)*(1-sigmoid(s)*self.wghts[layer+1][i][neuron])
                arr.append(ta)
            return np.sum(arr)
    
    def __get_all_partials__(self, y, ystar):
        res = []
        l1 = []
        for j in range(1, self.h_size):
            neur = []
            for k in range(0, self.n_features+1):
                neur.append(self.__get_partial_weight__(1, j, k, y, ystar))
            l1.append(neur)
        res.append(l1)
        l2 = []
        for j in range(1, self.h_size):
            neur = []
            for k in range(0, self.h_size):
                neur.append(self.__get_partial_weight__(2, j, k, y, ystar))
            l2.append(neur)
        res.append(l2)
        l3 = []
        neur = []
        for k in range(self.h_size):
            neur.append(self.__get_partial_weight__(3, 0, k, y, ystar))
        l3.append(neur)
        res.append(l3)


        return res



    def get_error(self, rows : DataFrame):
        #print(rows.head())
        #rows.apply(lambda row : print(f'{row.iloc[-1]}, {self.predict_from_sign(np.dot(row[:-1].to_numpy(), self.vec))}'), axis=1)
        correct = rows.apply(lambda row : (row.iloc[-1] == np.sign(self.predict(row[:-1]))), axis=1).sum()
        return 1-correct/len(rows)
            





        
        
        
def process_data(csvpath):
    train = pd.read_csv(csvpath + '/train.csv')   
    test = pd.read_csv(csvpath + '/test.csv') 
    #train.insert(len(train.columns)-1, 'ones', 1)
    #test.insert(len(test.columns)-1, 'ones', 1)  
    train['y'] = train['y'].replace(0, -1)
    test['y'] = test['y'].replace(0, -1)
    print(train.head())
    c = ThreeNeuralNetwork(100, len(train.columns) - 1, train)
    print(c.get_error(train))
    print(c.get_error(test))
    

process_data('../TestingDataBankNote')
'''
NN = ThreeNeuralNetwork(3, 4, None)
NN.vec = np.array([1, 1, 1, 1, 1])
print(NN.__get_neuron_value__(3, 1))
print(NN.__get_all_partials__(23423, 3))
'''