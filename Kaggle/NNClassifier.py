from numpy.core.arrayprint import DatetimeFormat
from pandas.core.frame import DataFrame

from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import optimizers
import keras_tuner as kt
import numpy as np
import pandas as pd
from tensorflow.keras.wrappers import scikit_learn


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
t = train.pop('education')
t = test.pop('education')
train = pd.get_dummies(train)
train['native.country_Holand-Netherlands'] = np.zeros(len(train))
scaler = MinMaxScaler()
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

def create_model():
    model = Sequential()
    model.add(Dense(31, input_dim=89, activation='relu' ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_optimizer(hp):
    model = Sequential()
    hp_units = hp.Int('units', min_value=10, max_value=200)
    #hp_units_2 = hp.Int('units2', min_value=10, max_value=200)
    model.add(Dense(units=hp_units, input_dim=89, activation='relu' ))
    #model.add(Dense(units=hp_units_2, activation='relu' ))
    model.add(Dense(1, activation='sigmoid'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return model

def model_optimizer_2(hp):
    model = Sequential()
    hp_units = hp.Int('units', min_value=10, max_value=200)
    hp_layers = hp.Int('layers', min_value=1, max_value=20)
    for i in range(hp_layers):
        model.add(Dense(units=hp_units, input_dim=89, activation='relu' ))
    model.add(Dense(1, activation='sigmoid'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return model


def print_predict(gbc):
    df = DataFrame(gbc.predict(test))
    print(df)
    df['ID'] = ID
    df = df.iloc[:, 0:]
    print(df.head())
    df.to_csv('out3.csv', index=False)



'''
n = len(train)
test = train.iloc[int(n*.9):]
val = train.iloc[int(n*.8//1):int(n*.9//1)]
train = train.iloc[:int(n*.8//1)]
trainhere = train
valhere = val
print(len(valhere))
print(valhere.iloc[:, -3:].head())
'''


'''
hyperparameter optimization
tuner = kt.Hyperband(model_optimizer,
                     objective='val_accuracy',
                     max_epochs=30,
                     factor=3,
                     directory='ktdir',
                     project_name='incomeprediction',
                     overwrite=False)
tuner.search(train.iloc[:, :-1], train.iloc[:, -1], epochs=20, validation_split=.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(train.iloc[:, :-1], train.iloc[:, -1], epochs=20, validation_split=.2)
'''
gbc = KerasClassifier(build_fn=create_model, epochs=10, batch_size=5, verbose=0)
gbc.fit(train.iloc[:, :-1], train.iloc[:, -1])
print_predict(gbc)
#kfold validation code --- 
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(gbc, train.iloc[:, :-1], train.iloc[:, -1], cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

'''
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))
print(valhere.head())
print(gbc.predict(valhere.iloc[:, :-1]))
print(gbc.score(valhere.iloc[:, :-1], valhere.iloc[:, -1]))
print(gbc.score(test.iloc[:, :-1], test.iloc[:, -1]))
'''

