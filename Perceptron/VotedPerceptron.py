import numpy as np
from numpy.core.arrayprint import DatetimeFormat
import pandas as pd
from pandas.core.frame import DataFrame

class PerceptronClassifier():
    
    def __init__(self, train : DataFrame, lrate = 1, epoch=10) -> None:
        self.vec = np.zeros(len(train.columns)-1)
        self.vecweights = []
        m=0
        where = 1
        for i in range(0, epoch):
            here = train.sample(frac=1)
            for i in range(0, len(train)):
                row = here.iloc[i]
                nprow = row[:-1].to_numpy()
                pred = np.sign(np.dot(self.vec, nprow))
                pred = 0 if pred == -1 else 1
                real = row[-1]
                if pred != real:
                    self.vecweights.append((self.vec, where))
                    self.vec = np.add(self.vec, ((real-pred)*lrate)*nprow)
                    where=1
                else:
                    where+=1
    
    def predict_correct(self, row):
        preds = [where*np.sign(np.dot(vechere, row[:-1].to_numpy())) for vechere, where in self.vecweights]
        pred = 0 if np.sign(sum(preds)) == -1 else 1
        return pred == row[-1]
    
    def get_error(self, rows : DataFrame):
        #print(rows.head())
        #rows.apply(lambda row : print(f'{row.iloc[-1]}, {self.predict_from_sign(np.dot(row[:-1].to_numpy(), self.vec))}'), axis=1)
        correct = rows.apply(lambda row : (row.iloc[-1] == self.predict_from_sign(np.dot(row[:-1].to_numpy(), self.vec))), axis=1).sum()
        return 1-correct/len(rows)

    def predict_from_sign(self, num):
        return 1 if np.sign(num) == 1 else 0


        







def process_data(csvpath):
    train = pd.read_csv(csvpath + '/train.csv')   
    train.insert(len(train.columns)-1, 'ones', 1)
    test = pd.read_csv(csvpath + '/train.csv')   
    test.insert(len(test.columns)-1, 'ones', 1)
    
    c = PerceptronClassifier(train)
    print(c.get_error(test))
    print(c.vecweights)
    

process_data('../TestingDataBankNote')