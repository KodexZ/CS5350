from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.function_base import _gradient_dispatcher
from pandas.core.frame import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from keras.layers import Dense
from keras.models import Sequential
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from skopt import BayesSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import optimizers
import skopt
import keras_tuner as kt
import numpy as np
import pandas as pd


def encode_categories(df : DataFrame):
    return pd.get_dummies(df)

def replace_missing(df : DataFrame, column : str):
    df[column] = df[column].fillna(df[column].mode()[0])
    return df

def print_predict(gbc):
    df = DataFrame(gbc.predict(test))
    print(df)
    df['ID'] = ID
    df = df.iloc[:, 0:]
    print(df.head())
    df.to_csv('out3.csv', index=False)
    print('done')

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
t = train.pop('education')
t = test.pop('education')
train = pd.get_dummies(train)
train['native.country_Holand-Netherlands'] = np.zeros(len(train))
scaler = StandardScaler()
scaled_features = scaler.fit_transform(train)
train = pd.DataFrame(scaled_features, index=train.index, columns=train.columns)

train.insert(len(train.columns),'income>50K', end)
print(train.iloc[:, -1].head())

test = pd.get_dummies(test)
ID = test.pop('ID')
ned = test.pop('native.country_Holand-Netherlands')
test.insert(len(test.columns), 'native.country_Holand-Netherlands', ned)
scaled_features_test = scaler.fit_transform(test)
test = pd.DataFrame(scaled_features_test, index=test.index, columns=test.columns)
print(len(train.columns))



def model_optimizer():
    params = dict()
    params['learning_rate'] = skopt.space.Real(.05, 10)
    params['n_estimators'] = skopt.space.Integer(10, 300)
    params['max_depth'] = skopt.space.Integer(1, 7)
    search = BayesSearchCV(estimator=GradientBoostingClassifier(), search_spaces=params, n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle=True))
    #model = GradientBoostingClassifier(learning_rate=hp_lrate, n_estimators=hp_estimators, max_depth=hp_max_depth)
    #hp_units_2 = hp.Int('units2', min_value=10, max_value=200)
    #model.add(Dense(units=hp_units_2, activation='relu' ))
    search.fit(train.iloc[:, :-1], train.iloc[:, -1])
    print(search.best_score_)
    print(search.best_params_)

gbc = GradientBoostingClassifier(learning_rate=.05, n_estimators=300, max_depth=5)

gbc.fit(train.iloc[:, :-1], train.iloc[:, -1])
print_predict(gbc)

n = len(train)
model_optimizer()
#kfold validation code --- 
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(gbc, train.iloc[:, :-1], train.iloc[:, -1], cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


