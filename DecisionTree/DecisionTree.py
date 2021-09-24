from typing import DefaultDict
from numpy import recarray
from pandas.core.frame import DataFrame
from scipy.stats import entropy
import pandas as pd
from dataclasses import dataclass
from typing import List
import numpy as np
import sys
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
        self.errors = [self.__get_entropy_column__, self.__Get_ME_column__, self.__get_GINI_column__]
        self.errorfunc = self.errors[error]
        self.data = self.data.rename(columns={self.data.columns[-1] : 'P'})
        self.m_depth = m_depth      
        self._most_common_on_rule__ = {}
        self.__construct_tree__(data)

    def __str__(self):
        return str(self.tree)

    def __get_entropy_column__(self, col : DataFrame):
        return entropy(pd.Series.value_counts(col).to_numpy())
    
    def __get_GINI_column__(self, col: DataFrame):
        datavec = normalize(pd.Series.value_counts(col).to_numpy(), norm='l1')
        return 1 - np.dot(datavec, np.transpose(datavec))

    def __Get_ME_column__(col : DataFrame):
        return (len(col) - len(col[col['P'] == col['P'].mode()[0]])) / len(col)

    def __split_on_column__(self, df : DataFrame, col) -> DataSplit:
        
        vals = list(set(df[col]))
        res = []
        gain = self.errorfunc(df['P'])
        for val in vals:
            split = df[df[col] == val]
            gain -= (len(split)/len(df))*self.errorfunc(split['P'])
            res.append(split)
        return DataSplit(res, gain, col, vals)

    def __get_attribute_and_split_data_highest_gain__(self, df : DataFrame):
        rec : DataSplit = DataSplit([], -1, '', [])
        for col in df.columns[:-1]:
            here = self.__split_on_column__(df, col)
            if here.gain > rec.gain:
                
                rec = here  
        return rec

    
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
                mode = data['P'].mode()[0]
                self._most_common_on_rule__[f'rule {ruleCounter}'] = mode          
                __construct_tree_helper__(f'rule {ruleCounter}', d2, mode, depth + 1)
        
        ruleCounter = 0
        self.tree = DefaultDict(dict)
        mode = data['P'].mode()[0]
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
    print('test')

    for i in range(1, 7):
        curTree = DecisionTree(train, i)
        for j in range(0, 3):
            curTree.errorfunc = curTree.errors[j]      
            print(f'Calculating prediction percent correct of Tree of size {i} using error ' + str(curTree.errorfunc.__name__).replace('_', ' ') + '...')
            correct = sum(row['P'] == curTree.predict(row) for i, row in test.iterrows())
            print(f'\\\\{correct/len(test)}\\\\')
                



def norm_to_med(df : DataFrame, med : float):
    return df.apply(lambda x : 0 if x < med else 1)
    

def preprocess_numerical_data(df : DataFrame):
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = norm_to_med(df[col], df[col].median())

def preprocess_unknown_data (df : DataFrame):
    for col in df.columns:
        df[col] = df[col].replace('unknown', df[col].mode()[0])



def test_bank_data():
    train = pd.read_csv('TestingDataBank/train.csv')
    train = train.rename(columns={train.columns[-1] : 'P'})
    test = pd.read_csv('TestingDataBank/test.csv')
    test = test.rename(columns={test.columns[-1] : 'P'})

    preprocess_numerical_data(train)
    preprocess_numerical_data(test)
    preprocess_unknown_data(train)
    preprocess_unknown_data(test)
    for i in range(1, 17):
        curTree = DecisionTree(train, i)
        for j in range(0, 3):
            curTree.errorfunc = curTree.errors[j]      
            print(f'Calculating prediction percent correct of Tree of size {i} using error ' + str(curTree.errorfunc.__name__).replace('_', ' ') + '...')
            correct = sum(row['P'] == curTree.predict(row) for i, row in test.iterrows())
            print(f'\\\\{correct/len(test)}\\\\')


test_bank_data()
#test_car_data()