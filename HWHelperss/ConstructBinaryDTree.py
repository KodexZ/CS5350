from typing import DefaultDict
from pandas.core.frame import DataFrame
from scipy.stats import entropy
import pandas as pd

def get_entropy_binary_column(col : DataFrame):
    ones = col.sum()
    zeroes = len(col) - ones
    e = entropy([ones, zeroes])
    return e

def get_attribute_highest_gain_binary(df : DataFrame):
    record, colrec = -1, -1
    for col in df.columns[:-1]:
        split0, split1 = df[df[col] == 0], df[df[col] == 1]
        l1, l2 = len(split0), len(split1)
        gain = get_entropy_binary_column(df['y']) - (l1*get_entropy_binary_column(split0['y']) + l2*get_entropy_binary_column(split1['y']))/len(df)
        if gain > record:
            record = gain
            colrec = col
    return colrec



    
def construct_tree():
    def construct_tree_helper(parent, data : DataFrame):
        if len(data) == 0:
            return
        nonlocal ruleCounter
        if data['y'].sum() == 0 or data['y'].sum() == len(data):
            tree[parent] = data['y'].iloc[0]
            return
        
        split_on = get_attribute_highest_gain_binary(data)
        ruleCounter+=1
        tree[parent][(split_on, 0)] = f'rule {ruleCounter}'
        construct_tree_helper(f'rule {ruleCounter}', data[data[split_on] == 0])
        
        ruleCounter += 1
        tree[parent][(split_on, 1)] = f'rule {ruleCounter}'      
        construct_tree_helper(f'rule {ruleCounter}', data[data[split_on] == 1])

    data = pd.DataFrame([
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0]
    ], columns=['x1', 'x2', 'x3', 'x4', 'y'])
    tree = DefaultDict(dict)
    ruleCounter=0
    construct_tree_helper(f'rule {ruleCounter}', data)
    print(tree)
    


construct_tree()