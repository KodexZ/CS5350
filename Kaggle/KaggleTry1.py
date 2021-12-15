from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd


train = pd.read_csv('./data/train_final.csv')
n = len(train)
test = train.iloc[int(n*.9):]
val = train.iloc[int(n*.8//1):int(n*.9//1)]
train = train.iloc[:int(n*.8//1)]
trainhere = train.loc[:, train.dtypes=='int64'].head()
valhere = val.loc[:, val.dtypes=='int64']
print(len(valhere))
gbc = GradientBoostingClassifier()
gbc.fit(trainhere.iloc[:, :-1], trainhere.iloc[:, -1])
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))
print(valhere.head())
print(gbc.predict(valhere.iloc[:, :-1]))
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))