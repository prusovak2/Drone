import pandas as pd
import numpy as np
import graphviz
from CreateDataMatrixForDT import dataForDTRealImagFrozenDict
from CreateDataMatrixForDT import Cmd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def GetLabel(labelColumn, dataMatrix):
    labels = dataMatrix[[labelColumn]]
    labels = labels.reset_index()
    labels = labels.drop(['time'], axis=1)
    # print(labels)
    # print(labels.describe())
    return labels

def GetFeatures(dataMatrix):
    features = dataMatrix.drop(['leftRight', 'frontBack', 'angular'], axis=1)
    return features

def CrossValidation(pipeline, params_to_try, x_train, y_train, x_test, y_test):
    score = 0
    DT_criterion = ''
    DT_maxDepth = 0
    for cv in range(3, 6):
        create_grid = GridSearchCV(pipeline, param_grid=params_to_try, cv=cv)
        create_grid.fit(x_train, y_train)
        newScore = create_grid.score(x_test, y_test)
        print("score for %d fold cross validation is %3.2f" % (cv, newScore))
        print("best fit params:")
        bestParams = create_grid.best_params_
        print(bestParams)
        if newScore > score:
            score = newScore
            DT_criterion = bestParams['decisionTree__criterion']
            DT_maxDepth = bestParams['decisionTree__max_depth']
    return {'DT_criterion': DT_criterion, 'DT_maxDepth': DT_maxDepth}

def BuildDT(labelColumn, dataMatrix, pipeline, params_to_try):
    # split what I wanna predict from what I wanna base a prediction on
    labels = GetLabel(labelColumn, dataMatrix)
    features = GetFeatures(dataMatrix)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, stratify=labels)
    print("num of training samples: ", len(x_train))
    print("num of test samples: ", len(y_test))

    # estimate the best params for DT
    bestParams = CrossValidation(pipeline, params_to_try, x_train, y_train, x_test, y_test)

    # build DT
    decisionTree = DecisionTreeClassifier(criterion=bestParams['DT_criterion'], max_depth=bestParams['DT_maxDepth'])
    decisionTree.fit(x_train, y_train)

    return decisionTree, x_train, y_train, x_test, y_test


def GraphTree(decisionTree, features, pngFileName):
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus
    from CreateDataMatrixForDT import frozenCmds

    dot_data = StringIO()

    print([*frozenCmds.values()])
    featureNames = features.columns
    export_graphviz(decisionTree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=featureNames,
                    class_names=[*frozenCmds.values()])

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('OutputStages\\Graphs\\%s' %pngFileName)
    Image(graph.create_png())


dataForDT = dataForDTRealImagFrozenDict
# prepare pipeline, that carries out a data standartization - to make a features have a variance in a same order
# and creates DT
pipe_steps = [('scaler', StandardScaler()), ('decisionTree', DecisionTreeClassifier())]
pipeline = Pipeline(pipe_steps)
print(pipeline)
print(pipeline.get_params().keys())

# DT attributes I wanna estimate - are there any other?
params_to_try = {'decisionTree__criterion': ['gini', 'entropy'],
                 'decisionTree__max_depth': np.arange(3, 15)}
# np.arange(3, 15): totally random estimation of max tree depth taken from a tutorial. I have no clue whether it
# makes sense in this case!!

# build and graph decision trees
features = GetFeatures(dataForDT)
DTleftRight, x_trainLR, y_trainLR, x_testLR, y_testLR = BuildDT('leftRight', dataForDT, pipeline, params_to_try)
GraphTree(DTleftRight, features, 'leftRightDT.png')
DTfrontBack, x_trainFB, y_trainFB, x_testFB, y_testFB = BuildDT('frontBack', dataForDT, pipeline, params_to_try)
GraphTree(DTfrontBack, features, 'frontBackDT.png')
DTangular, x_trainA, y_trainA, x_testA, y_testA = BuildDT('angular', dataForDT, pipeline, params_to_try)
GraphTree(DTangular, features, 'angular.png')







