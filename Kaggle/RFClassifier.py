from numpy.core.arrayprint import DatetimeFormat
from pandas.core.frame import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd


def encode_categories(df : DataFrame):
    return pd.get_dummies(df)

def replace_missing(df : DataFrame, column : str):
    df[column] = df[column].fillna(df[column].mode()[0])
    return df
train = pd.read_csv('./data/train_final.csv')
test = pd.read_csv('./data/test_final.csv')
test['native.country'] = test['native.country'].replace('?',np.nan)
test['workclass'] = test['workclass'].replace('?',np.nan)
test['occupation'] = test['occupation'].replace('?',np.nan)
train['native.country'] = train['native.country'].replace('?',np.nan)
train['workclass'] = train['workclass'].replace('?',np.nan)
train['occupation'] = train['occupation'].replace('?',np.nan)
#train = replace_missing(train, 'native.country')
#train = replace_missing(train, 'workclass')
#train = replace_missing(train, 'occupation')
#train.dropna(how='any',inplace=True)
end = train.pop('income>50K')
#t = train.pop('education')
#t = test.pop('education')
train = pd.get_dummies(train)
train['native.country_Holand-Netherlands'] = np.zeros(len(train))
train.insert(len(train.columns),'income>50K', end)


test = pd.get_dummies(test)
ID = test.pop('ID')
ned = test.pop('native.country_Holand-Netherlands')
test.insert(len(test.columns), 'native.country_Holand-Netherlands', ned)


gbc = RandomForestClassifier()

gbc.fit(train.iloc[:, :-1], train.iloc[:, -1])
print(len(train.columns))
print(len(test.columns))
df = DataFrame(gbc.predict(test))
print(df)
df['ID'] = ID
df = df.iloc[:, 0:]
print(df.head())
df.to_csv('out3.csv', index=False)


n = len(train)
test = train.iloc[int(n*.9):]
val = train.iloc[int(n*.8//1):int(n*.9//1)]
train = train.iloc[:int(n*.8//1)]
trainhere = train
valhere = val
print(len(valhere))
print(valhere.iloc[:, -3:].head())

gbc.fit(trainhere.iloc[:, :-1], trainhere.iloc[:, -1])
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))
print(valhere.head())
print(gbc.predict(valhere.iloc[:, :-1]))
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))
print(gbc.score(test.iloc[:, :-1], test.iloc[:, -1]))

