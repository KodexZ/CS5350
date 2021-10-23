
import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import to_time
from sklearn import linear_model




def concrete_batch_gradient_regression(csvpath : str, learning_rate : float, start : np.ndarray, itrs=30000):
     n = len(start)
     reg = linear_model.LinearRegression()
     

     def grad(i):
         row = train.sample(1).iloc[0]
         ehere = np.dot(cur, row[:-1].to_numpy())
         res = [(row.iloc[-1] - ehere)*row.iloc[j] for j in range(n)]
         return res
             
     train = pd.read_csv(csvpath+'train.csv')
     test = pd.read_csv(csvpath+'test.csv')
     #print(train.columns)
     cur = start
     print(cur)
     for i in range(itrs):

         cur = np.add(cur, np.multiply(learning_rate, grad(i)))
         #print(cur)
         print(train.apply(lambda row : (row.iloc[-1] - np.dot(row[:-1].to_numpy(), cur))**2, axis=1).sum())
     print(cur)
     print(f'Final test error is {test.apply(lambda row : (row.iloc[-1] - np.dot(row[:-1].to_numpy(), cur))**2, axis=1).sum()}')
     





if __name__ == '__main__':
    import os
    print(os.path.dirname(os.path.realpath(__file__)))
    
    #test_bank_data()
    print(concrete_batch_gradient_regression('C:/Users/korto/School/CS5350/TestingDataConcrete/', .001, np.zeros(7)))
