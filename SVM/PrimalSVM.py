import numpy as np
from numpy.core.arrayprint import DatetimeFormat
import pandas as pd
from pandas.core.frame import DataFrame

class SVMClassifier():
    
    def __init__(self, train : DataFrame, lrate = .02, epoch=50, C=700/873, alpha=20) -> None:
        self.vec = np.zeros(len(train.columns)-1)
        self.olrate = lrate
        self.alpha = 1
        n = len(train)
        for i in range(0, epoch):
            here = train.sample(frac=1)
            for i in range(0, len(train)):
                row = here.iloc[i]
                actual = row[-1]
                if actual == 0:
                    actual = -1
                nprow = row[:-1].to_numpy()
                if actual*np.dot(self.vec, nprow) <= 1:
                    w0 = self.vec   
                    w0[-1] = 0
                    self.vec = np.add(self.vec, -lrate*w0)
                    self.vec = np.add(self.vec, lrate*C*n*actual*nprow)
                else:
                    oldlast = nprow[-1]
                    self.vec = (1-lrate)*self.vec
                    self.vec[-1]=oldlast                   
            lrate = self.get_next_lrate_0(lrate, i)
    def predict_correct(self, row):
        pred = np.sign(np.dot(self.vec, row[:-1].to_numpy()))
        pred = 0 if pred == -1 else 1
        return pred == row[-1]
    
    def get_error(self, rows : DataFrame):
        #print(rows.head())
        #rows.apply(lambda row : print(f'{row.iloc[-1]}, {self.predict_from_sign(np.dot(row[:-1].to_numpy(), self.vec))}'), axis=1)
        correct = rows.apply(lambda row : (row.iloc[-1] == self.predict_from_sign(np.dot(row[:-1].to_numpy(), self.vec))), axis=1).sum()
        return 1-correct/len(rows)

    def predict_from_sign(self, num):
        return 1 if np.sign(num) == 1 else 0

    def get_next_lrate_0(self, lrate, t):
        return self.olrate/(1+t)
        


def process_data(csvpath):
    train = pd.read_csv(csvpath + '/train.csv')   
    train.insert(len(train.columns)-1, 'ones', 1)
    test = pd.read_csv(csvpath + '/test.csv')   
    test.insert(len(test.columns)-1, 'ones', 1)
    print(train.head())
    c = SVMClassifier(train)
    print(c.get_error(train))
    print(c.get_error(test))
    print(c.vec)
    

process_data('../TestingDataBankNote')