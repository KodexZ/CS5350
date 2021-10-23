from typing import DefaultDict
from numpy import recarray
from pandas.core.frame import DataFrame
from scipy.stats import entropy
import pandas as pd
from dataclasses import dataclass
from typing import List
import numpy as np
import sys
from scipy import stats
from sklearn.preprocessing import normalize
from pandas.api.types import is_numeric_dtype
sys.path.append('TestData')

@dataclass
class DataSplit:
        datas : List[DataFrame]
        gain : float
        attribute : str
        vals : List[str]

class DecisionTree:
    def __init__(self, data : DataFrame, m_depth=5, error=0):
        self.data = data
        self.errors = [self.__get_weighted_entropy_column__]
        self.errorfunc = self.errors[error]
        self.data = self.data.rename(columns={self.data.columns[-1] : 'P'})
        self.m_depth = m_depth      
        self._most_common_on_rule__ = {}
        self.__construct_tree__(data)

    def __str__(self):
        return str(self.tree)

    def __get_weighted_entropy_column__(self, df : DataFrame, col : str):
        #return entropy(pd.Series.value_counts(col).to_numpy())
        #We need to redefine entropy to take weights into account
        vals = df[col].unique()
        entrlst = []
        for val in vals:
            entrlst.append(df[df[col] == val]['W'].sum())
        return entropy(entrlst)

    def __split_on_column__(self, df : DataFrame, col) -> DataSplit:
        #change to df[col].unique()
        #we change to pass entire dataframe instead of just the column and use the weights
        vals = list(set(df[col]))
        res = []
        gain = self.errorfunc(df, 'P')
        for val in vals:
            split = df[df[col] == val]
            gain -= (len(split)/len(df))*self.errorfunc(split, 'P')
            res.append(split)
        return DataSplit(res, gain, col, vals)

    def __get_attribute_and_split_data_highest_gain__(self, df : DataFrame):
        rec : DataSplit = DataSplit([], -1, '', [])
        for col in df.columns[:-2]:
            here = self.__split_on_column__(df, col)
            if here.gain > rec.gain:
                
                rec = here  
        return rec


    def __get_weighted_mode__(self, df : DataFrame, col):
        vals = df[col].unique()
        rec = -float('inf')
        recm = '--'
        for val in vals:
            here = df[df[col] == val]['W'].sum()
            if here > rec:
                rec=here
                recm = val
        return recm
    
    def __construct_tree__(self, data):  
        def __construct_tree_helper__(parent, data : DataFrame, mostcommonlast, depth : int):
            nonlocal ruleCounter
            if depth > self.m_depth:
                ruleCounter+=1
                self.tree[parent] = mostcommonlast
                return       
            
            if len(data) == 0:
                ruleCounter+=1
                self.tree[parent] = mostcommonlast
                return
            if data['P'].sum() == 0 or data['P'].sum() == len(data):
                self.tree[parent] = data['P'].iloc[0]
                return
            
            split_on = self.__get_attribute_and_split_data_highest_gain__(data)
            
            for d2, on in zip(split_on.datas, split_on.vals):
                ruleCounter+=1
                self.tree[parent][(split_on.attribute, on)] = f'rule {ruleCounter}'
                mode = self.__get_weighted_mode__(data, 'P')
                self._most_common_on_rule__[f'rule {ruleCounter}'] = mode          
                __construct_tree_helper__(f'rule {ruleCounter}', d2, mode, depth + 1)
        
        ruleCounter = 0
        self.tree = DefaultDict(dict)
        mode = self.__get_weighted_mode__(data, 'P')
        self._most_common_on_rule__[f'rule {ruleCounter}'] = mode
        __construct_tree_helper__(f'rule {ruleCounter}', data, mode, 0)
        

    def predict(self, row):       
       cur = 'rule 0'
       #While we have a tuple (rule), follow that rule
       while type(self.tree[cur]) is dict:
           #grab the value we're splitting on -- honestly refactor this code later this is a messy way of storing it
            val = next(iter(self.tree[cur].keys()))[0]
            
            #If a branch exists for the given information, take the branch, else return the most popular value at this point
            if (val, row[val]) in self.tree[cur]:
                cur = self.tree[cur][(val, row[val])]
            else:
                return self._most_common_on_rule__[cur]
                
                

           
       return self.tree[cur]
        
        
def test_car_data():
    train = pd.read_csv('TestingDataCar/train.csv')
    train = train.rename(columns={train.columns[-1] : 'P'})
    test = pd.read_csv('TestingDataCar/test.csv')
    test = test.rename(columns={test.columns[-1] : 'P'})
    train.insert(len(train.columns) - 1, 'W', np.ones(len(train)))
    test.insert(len(test.columns) - 1, 'W', np.ones(len(test)))

    for i in range(1, 7):
        curTree = DecisionTree(train, i)
        for j in range(0, 1):
            curTree.errorfunc = curTree.errors[j]      
            print(f'Calculating prediction percent correct of Tree of size {i} using error ' + str(curTree.errorfunc.__name__).replace('_', ' ') + '...')
            correct = sum(row['P'] == curTree.predict(row) for i, row in test.iterrows())
            print(f'\\\\{correct/len(test)}\\\\')
                



def norm_to_med(df : DataFrame, med : float):
    return df.apply(lambda x : 0 if x < med else 1)
    

def preprocess_numerical_data(df : DataFrame):
    for col in df.columns:
        if is_numeric_dtype(df[col]) and col != 'W' and col != 'P':
            df[col] = norm_to_med(df[col], df[col].median())

def preprocess_unknown_data (df : DataFrame):
    for col in df.columns:
        df[col] = df[col].replace('unknown', df[col].mode()[0])



def test_bank_data():
    train = pd.read_csv('TestingDataBank/train.csv')
    train = train.rename(columns={train.columns[-1] : 'P'})
    test = pd.read_csv('TestingDataBank/test.csv')
    test = test.rename(columns={test.columns[-1] : 'P'})
    train.insert(len(train.columns) - 1, 'W', np.ones(len(train)))
    test.insert(len(test.columns) - 1, 'W', np.ones(len(test)))
    print(test.head())
    preprocess_numerical_data(train)
    preprocess_numerical_data(test)
    for i in range(1, 17):
        curTree = DecisionTree(train, i)
        for j in range(0, 1):
            curTree.errorfunc = curTree.errors[j]      
            print(f'Calculating prediction percent correct of Tree of size {i} using error ' + str(curTree.errorfunc.__name__).replace('_', ' ') + '...')
            correct = sum(row['P'] == curTree.predict(row) for i, row in test.iterrows())
            print(f'\\\\{correct/len(test)}\\\\')


def ADABoostError(csvpath : str, itrs : int):
    
    depth = 1
    train = pd.read_csv(csvpath+'train.csv')
    train = train.rename(columns={train.columns[-1] : 'P'})
    test = pd.read_csv(csvpath+'test.csv')
    test = test.rename(columns={test.columns[-1] : 'P'})
    train = train.replace('no', -1)
    train = train.replace('yes', 1)
    test = test.replace('no', -1)
    test = test.replace('yes', 1)
    print(train.head())
    train.insert(len(train.columns) - 1, 'W', np.ones(len(train)))
    preprocess_numerical_data(train)
    preprocess_numerical_data(test)
    trees = []
    firstTree = DecisionTree(train, depth)
    correct = sum(row['P'] == firstTree.predict(row) for i, row in train.iterrows())
    lastError = 1 - correct/len(train)
    trees.append(firstTree)
    alphas = [1]
    curTree = firstTree
    errors = []
    testErrors = []
    totErrors = []
    totTestErrors = []
    
    for i in range(0, itrs-1):
        
        alpha = (1/2)*np.log((1-lastError)/lastError)
        alphas.append(alpha)
        for i, row in train.iterrows():
            sgn = -1 if row['P'] == curTree.predict(row) else 1
            if lastError == 0 or lastError == 1:
                print('WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
            
            #print(train.iloc[i]['W'])
            new = np.exp(alpha*sgn)*train.at[i, 'W']
            train.at[i,'W'] = new
            if train.at[i,'W'] == 0:
                print('imp')
            #print(train.iloc[i]['W'])
        curTree = DecisionTree(train, depth) 
        correct = sum(row['P'] == curTree.predict(row) for i, row in train.iterrows())
        correctt = sum(row['P'] == curTree.predict(row) for i, row in test.iterrows())
        correctthere = 0
        correcthere = 0
        for i, testrow in test.iterrows():
            lst = [tree.predict(testrow)*alphas[i] for i, tree in enumerate(trees)]
            pred = np.sign(sum(lst))
            correctthere += pred == testrow['P']
        totheretest = 1 - correctthere/len(test)

        for i, testrow in train.iterrows():
            lst = [tree.predict(testrow)*alphas[i] for i, tree in enumerate(trees)]
            pred = np.sign(sum(lst))
            correcthere += pred == testrow['P']
        totheretrain = 1 - correcthere/len(train)      
        lastError = 1 - correct/len(train)
        testLastError = 1 - correctt/len(test)

        errors.append(lastError)
        testErrors.append(testLastError)
        totErrors.append(totheretrain)
        totTestErrors.append(totheretest)
        print(f'{lastError}, {testLastError}, {totheretrain}, {totheretest}')
        trees.append(curTree)
    correct = 0

    #test after train
    print('\n'.join(str(error) for error in errors))
    print('endtesterr')
    print('\n'.join(str(error) for error in testErrors))
    print('endtesterr')
    print('\n'.join(str(error) for error in totErrors))
    print('endtesterr')
    print('\n'.join(str(error) for error in totTestErrors))
    print('endtesterr')

        
        

        

            
if __name__ == '__main__':
    import os
    print(os.path.dirname(os.path.realpath(__file__)))

    #test_bank_data()
    print(ADABoostError('C:/Users/korto/School/CS5350/TestingDataBank/', 100))

            
        
