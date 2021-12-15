from pandas.core.frame import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def encode_categories(df : DataFrame):
    return pd.get_dummies(df)

def replace_missing(df : DataFrame, column : str):
    df[column] = df[column].fillna(df[column].mode()[0])
    return df
train = pd.read_csv('./data/train_final.csv')
train['native.country'] = train['native.country'].replace('?',np.nan)
train['workclass'] = train['workclass'].replace('?',np.nan)
train['occupation'] = train['occupation'].replace('?',np.nan)
#train = replace_missing(train, 'native.country')
#train = replace_missing(train, 'workclass')
#train = replace_missing(train, 'occupation')
#train.dropna(how='any',inplace=True)
n = len(train)

end = train.pop('income>50K')
train.pop('education')
train = pd.get_dummies(train)
train.insert(len(train.columns),'income>50K', end)
train = train.sample(frac=1)

test = train.iloc[int(n*.9):]
val = train.iloc[int(n*.8//1):int(n*.9//1)]
train = train.iloc[:int(n*.8//1)]
trainhere = train
valhere = val
print(len(valhere))
print(valhere.iloc[:, -3:].head())

gbc = KNeighborsClassifier()
gbc.fit(trainhere.iloc[:, :-1], trainhere.iloc[:, -1])
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))
print(valhere.head())
print(gbc.predict(valhere.iloc[:, :-1]))
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))
print(gbc.score(test.iloc[:, :-1], test.iloc[:, -1]))
