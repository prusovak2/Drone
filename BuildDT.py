import pandas as pd
import numpy as np
from PrepareDataDT import dataForDTRealImagFrozenDict
from PrepareDataDT import Cmd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

dataForDTRealImagFrozenDict.describe()
dataForDT = dataForDTRealImagFrozenDict

# split what I wanna predict from what I wanna base a prediction on
labels = dataForDT[['leftRight']]

labels = labels.reset_index()
labels = labels.drop(['time'], axis=1)
print(labels)
print(labels.describe())
features = dataForDT.drop(['leftRight', 'frontBack', 'angular'], axis=1)

# save a part of data as a test data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, stratify=labels)

print("num of training samples: ", len(x_train))
print("num of test samples: ", len(y_test))

# prepare pipeline, that carries out a data standartization - to make a features have a variance in a same order
# and creates DT
pipe_steps = [('scaler', StandardScaler()), ('decisionTree', DecisionTreeClassifier())]
pipeline = Pipeline(pipe_steps)
print(pipeline)

# DT attributes I wanna estimate - are there any other
params_to_try = {'decisionTree__criterion': ['gini', 'entropy'],
                 'decisionTree__max_depth': np.arange(3, 15)}
# np.arange(3, 15): totally random estimation of max tree depth taken from a tutorial. I have no clue whether it
# makes sense in this case!!

print(pipeline.get_params().keys())

'''
for cv in range(3, 6):
    create_grid = GridSearchCV(pipeline, param_grid=params_to_try, cv=cv)
    create_grid.fit(x_train, y_train)
    print("score for %d fold cross validation is %3.2f", (cv, create_grid.score(x_test, y_test)))
    print("bast fit params:")
    print(create_grid.best_params_)
'''

decisionTree = DecisionTreeClassifier(criterion='gini', max_depth=6)
decisionTree.fit(x_train, y_train)

