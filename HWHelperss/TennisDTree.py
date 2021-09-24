from typing import DefaultDict
from numpy import recarray
from pandas.core.frame import DataFrame
from pandas.io.pytables import DataCol
from scipy.stats import entropy
import pandas as pd
from dataclasses import dataclass
from typing import List
import numpy as np
@dataclass

class DataSplit:
    datas : List[DataFrame]
    gain : float
    attribute : str
    vals : List[str]

def get_entropy_binary_column(col : DataFrame):
    ones = col.sum()
    zeroes = len(col) - ones
    e = entropy([ones, zeroes])
    return e

def get_ME_binary_column(col : DataFrame):
    ones = col.sum()
    zeroes = len(col) - ones
    tot = ones + zeroes
    e = min(tot - ones, tot - zeroes)/tot
    return e

def get_GINI_binary_column(col : DataFrame):
    ones = col.sum()
    zeroes = len(col) - ones 
    tot = ones + zeroes
    e = 1-((ones/tot)**2 + (zeroes/tot)**2)
    return e


def split_on_column(df : DataFrame, col) -> DataSplit:
    errorfunc = get_entropy_binary_column # choose error function
    
    vals = list(set(df[col]))
    res = []
    gain = errorfunc(df['P'])
    for val in vals:
        split = df[df[col] == val]
        gain -= (len(split)/len(df))*errorfunc(split['P'])
        res.append(split)
    return DataSplit(res, gain, col, vals)



def get_attribute_and_split_data_highest_gain(df : DataFrame):
    rec : DataSplit = DataSplit([], -1, '', [])
    for col in df.columns[:-1]:
        here = split_on_column(df, col)
        print(f'{col} : {here.gain}')
        if here.gain > rec.gain:
            
            rec = here  
    return rec

  
def construct_tree():  
    def construct_tree_helper(parent, data : DataFrame, mostcommonlast):       
        nonlocal ruleCounter
        if len(data) == 0:
            ruleCounter+=1
            tree[parent] = mostcommonlast
            return
        if data['P'].sum() == 0 or data['P'].sum() == len(data):
            tree[parent] = data['P'].iloc[0]
            return
        
        split_on = get_attribute_and_split_data_highest_gain(data)
        
        for d2, on in zip(split_on.datas, split_on.vals):
            ruleCounter+=1
            tree[parent][(split_on.attribute, on)] = f'rule {ruleCounter}'           
            construct_tree_helper(f'rule {ruleCounter}', d2, data['P'].mode())

    data = pd.DataFrame([
    ['S', 'H', 'H', 'W', 0],
    ['S', 'H', 'H', 'S', 0],
    ['O', 'H', 'H', 'W', 1],
    ['R', 'M', 'H', 'W', 1],
    ['R', 'C', 'N', 'W', 1],
    ['R', 'C', 'N', 'S', 0],
    ['O', 'C', 'N', 'S', 1],
    ['S', 'M', 'H', 'W', 0],
    ['S', 'C', 'N', 'W', 1],
    ['R', 'M', 'N', 'W', 1],
    ['S', 'M', 'N', 'S', 1],
    ['O', 'M', 'H', 'S', 1],
    ['O', 'H', 'N', 'W', 1],
    ['R', 'M', 'H', 'S', 0],
    ['O', 'M', 'N', 'W', 1]
    ], columns=['O', 'T', 'H', 'W', 'P'])
    ruleCounter = 0
    tree = DefaultDict(dict)
    #construct_tree_helper(f'rule {ruleCounter}', data, data['P'].mode())
    get_attribute_and_split_data_highest_gain(data)
    

    


construct_tree()