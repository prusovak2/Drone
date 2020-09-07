from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from BuildDT import GetLabel
from BuildDT import GetFeatures
from CreateDataMatrixForDT import dataForDTRealImagFrozenDict as dataForRF
from ConfusionMatrix import CreateConfusionMatrix
from CreateDataMatrixForDT import dataForCM
"""
rf = RandomForestClassifier(random_state = 42)
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
"""


def GetBestParamsRandomSearch(x_train, y_train):
	# Use the random grid to search for best hyperparameters
	# First create the base model to tune
	rf = RandomForestClassifier()

	# CREATE A GRID TO CHOOSE PARAMS FROM
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf,
				   'bootstrap': bootstrap}
	print("RANDOM GRID")
	pprint(random_grid)

	# Random search of parameters, using 3 fold cross validation,
	# search across 100 different combinations, and use all available cores
	rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=73, n_jobs=-1)

	# Fit the random search model
	FitRF(x_train, y_train, rf_random)

	print("BEST PARAMS")
	pprint(rf_random.best_params_)
	return rf_random.best_estimator_


def GetBestParamsGridSearch(paramGridBasedOnRandomSearch, labelColumnName, dataMatrix):
	grid_search = GridSearchCV(estimator=rf, param_grid=paramGridBasedOnRandomSearch,
							   cv=3, n_jobs=-1, verbose=2)
	grid_search = FitRF(labelColumnName, dataMatrix, grid_search)
	pprint(grid_search.best_params_)
	return grid_search.best_estimator_

def TrainTestSplit(labelColumn, dataMatrix):
	labels = GetLabel(labelColumn, dataMatrix)
	features = GetFeatures(dataMatrix)
	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, stratify=labels,
														random_state=73)
	return x_train, x_test, y_train, y_test


def GetLabelAndFeatureData(labelColumnName, dataMatrix):
	labels = GetLabel(labelColumnName, dataMatrix)
	features = GetFeatures(dataMatrix)
	y = labels[:]
	X = features.values
	return X, y


def FitRF(x_train, y_train, forest):
	y_train = np.ravel(y_train)
	forest.fit(x_train, y_train)
	return forest


if __name__ == "__main__":
	# ########################################## LEFT RIGHT ##################################
	# estimate some value of hyperparameters
	X, y = GetLabelAndFeatureData('leftRight', dataForRF)
	#bestEstimatorRandom = GetBestParamsRandomSearch(X, y)
	"""
	BEST PARAMS PROVIDED BY RANDOM SEARCH
	{'bootstrap': True,
	 'max_depth': 30,
	 'max_features': 'auto',
	 'min_samples_leaf': 2,
	 'min_samples_split': 2,
	 'n_estimators': 800}
	"""
	RandomRandomForest = RandomForestClassifier(n_estimators=800, bootstrap=True, max_depth=30, max_features='auto',
									min_samples_leaf=2, min_samples_split=2)
	RandomRandomForest = FitRF(X, y, RandomRandomForest)
	CreateConfusionMatrix('leftRight', dataForCM, RandomRandomForest, "BestEstimatorRandom", plt.cm.Reds)

	# create random forest with default parameters for comparison
	basicForest = RandomForestClassifier()
	basicForest = FitRF(X, y, basicForest)
	CreateConfusionMatrix('leftRight', dataForCM, basicForest, "BasicForestMatrix", plt.cm.Reds)

	# param grid based on params estimated by random search
	param_grid = {
		'bootstrap': [True],
		'max_depth': [20, 30, 40],
		'max_features': [2, 3, 4],
		'min_samples_leaf': [1, 2, 3],
		'min_samples_split': [2, 3, 5],
		'n_estimators': [300, 400, 600, 800, 1000]
	}

	# use GridSearchCV to determine a better hyperparameters based on params obtained by random search
	rf = RandomForestClassifier()
	#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,cv=3, n_jobs=-1, verbose=2)
	#grid_search = FitRF(X,y,grid_search)
	#pprint(grid_search.best_params_)
	"""
	BEST PARAMS PROVIDED BY GRID SEARCH
	{'bootstrap': True,
	 'max_depth': 30,
	 'max_features': 3,
	 'min_samples_leaf': 2,
	 'min_samples_split': 3,
	 'n_estimators': 400}
	"""
	# create RANDOM FOREST WITH THE BEST VERSION OF PARAMETERS FOR LEFT RIGHT
	gridSearchRandomForest = RandomForestClassifier(n_estimators=400, bootstrap=True, max_depth=30, max_features=3,
									min_samples_leaf=2, min_samples_split=3)
	gridSearchRandomForest = FitRF(X,y,gridSearchRandomForest)
	CreateConfusionMatrix('leftRight', dataForCM, gridSearchRandomForest, "GridSearchForestMatrix", plt.cm.Reds)



"""
forest = RandomForestClassifier(n_estimators=400, bootstrap=True, max_depth=30, max_features='sqrt',
									min_samples_leaf=1, min_samples_split=5)
forest = FitRF('leftRight', dataForRF, forest)
CreateConfusionMatrix('leftRight', dataForCM, forest, "randomForestMatrix", plt.cm.Reds)

basicForest = RandomForestClassifier()
basicForest = FitRF('leftRight', dataForRF, basicForest)
CreateConfusionMatrix('leftRight', dataForCM, basicForest, "BasicForestMatrix", plt.cm.Reds)

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [20, 30, 40],
    'max_features': [2,3,4],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [3,5,7],
    'n_estimators': [100, 200, 300, 1000]
}# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                          cv = 3, n_jobs=-1, verbose = 2)

CreateConfusionMatrix('leftRight', dataForCM, grid_search.best_estimator_, "GridSearchForestMatrix", plt.cm.Reds)
"""



"""
BEST PARAMS
{'bootstrap': True,
 'max_depth': 30,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 5,
 'n_estimators': 400}
"""
"""
BEST PARAMS
{'bootstrap': False,
 'max_depth': 90,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 10,
 'n_estimators': 800}
"""
