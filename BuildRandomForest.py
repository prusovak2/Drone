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
from CreateDataMatrix import dataForDTRealImagFrozenDict as dataForRF
from ConfusionMatrix import CreateConfusionMatrix
from CreateDataMatrix import dataForCM

# given dataMatrix created in CreateDataMatrixForDT modul, this modul attempts to tune hyperparameters for a RandomForest
# and subsequently creates RandomForest with the best hyperparameters found

def GetBestParamsRandomSearch(x_train, y_train):
	"""
	RandomizedSearchCV to estimate the best hyperparameter combination for a RandomForestClassifier
	:param x_train: train features
	:param y_train: train labels
	:return: RandomForest with the best hyperparameters found
	"""
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
	with open('OutputStages\\randomParams.txt', 'w') as f:
		pprint(random_grid, stream=f)

	# Random search of parameters, using 3 fold cross validation,
	# search across 100 different combinations, and use all available cores
	rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=73, n_jobs=-1)

	# Fit the random search model
	FitRF(x_train, y_train, rf_random)

	print("BEST PARAMS")
	pprint(rf_random.best_params_)
	return rf_random.best_estimator_


def GetBestParamsGridSearch(paramGridBasedOnRandomSearch, x_train, y_train):
	"""
	provided a param grid based on the best hyperparameters found by GetBestParamsRandomSearch tries all possible combinations
	of hyperparametres and finds the best one, uses 3 fold cross validation
	:param paramGridBasedOnRandomSearch:
	:param x_train:
	:param y_train:
	:return: RandomForest with the best hyperparameters found
	"""
	grid_search = GridSearchCV(estimator=rf, param_grid=paramGridBasedOnRandomSearch,
							   cv=3, n_jobs=-1, verbose=2)
	grid_search = FitRF(x_train, y_train, grid_search)
	pprint(grid_search.best_params_)
	return grid_search.best_estimator_

def TrainTestSplit(labelColumn, dataMatrix):
	"""
	splits data from data matrix into train features, train labels, test features and test labels
	:param labelColumn:
	:param dataMatrix:
	:return:
	"""
	labels = GetLabel(labelColumn, dataMatrix)
	features = GetFeatures(dataMatrix)
	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, stratify=labels,
														random_state=73)
	return x_train, x_test, y_train, y_test


def GetLabelAndFeatureData(labelColumnName, dataMatrix, time=True):
	"""
	splits data from dataMatrix into features and labels
	:param labelColumnName:
	:param dataMatrix:
	:return:
	"""
	labels = GetLabel(labelColumnName, dataMatrix, time)
	features = GetFeatures(dataMatrix)
	y = labels[:]
	X = features.values
	return X, y


def FitRF(x_train, y_train, forest):
	"""
	trains given estimator on given data
	:param x_train: train features
	:param y_train: train labels
	:param forest: estimator to train
	:return: trained estimator
	"""
	y_train = np.ravel(y_train)
	forest.fit(x_train, y_train)
	return forest


if __name__ == "__main__":
	# as bestEstimatorRandom and GetBestParamsGridSearch methods run too long, their return values are hardcoded here
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
	CreateConfusionMatrix('leftRight', dataForCM, RandomRandomForest, "BestEstimatorRandomLR", plt.cm.Reds)

	# create random forest with default parameters for comparison
	basicForest = RandomForestClassifier()
	basicForest = FitRF(X, y, basicForest)
	CreateConfusionMatrix('leftRight', dataForCM, basicForest, "BasicForestMatrixLR", plt.cm.Reds)

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
	CreateConfusionMatrix('leftRight', dataForCM, gridSearchRandomForest, "GridSearchForestMatrixLeftRight", plt.cm.Reds)

	# ########################################## FRONT BACK ##################################
	# estimate some value of hyperparameters
	X, y = GetLabelAndFeatureData('frontBack', dataForRF)
	#bestEstimatorRandom = GetBestParamsRandomSearch(X, y)
	"""
	BEST PARAMS PROVIDED BY RANDOM SEARCH
	{'bootstrap': False,
	 'max_depth': 20,
	 'max_features': 'auto',
	 'min_samples_leaf': 2,
	 'min_samples_split': 5,
	 'n_estimators': 2000}
	"""
	RandomRandomForest = RandomForestClassifier(n_estimators=2000, bootstrap=False, max_depth=20, max_features='auto',
									min_samples_leaf=2, min_samples_split=5)
	RandomRandomForest = FitRF(X, y, RandomRandomForest)
	CreateConfusionMatrix('frontBack', dataForCM, RandomRandomForest, "BestEstimatorRandomFB", plt.cm.Reds)

	# create random forest with default parameters for comparison
	basicForest = RandomForestClassifier()
	basicForest = FitRF(X, y, basicForest)
	CreateConfusionMatrix('frontBack', dataForCM, basicForest, "BasicForestMatrixFB", plt.cm.Reds)

	# param grid based on params estimated by random search
	param_grid = {
		'bootstrap': [False],
		'max_depth': [10, 20, 30],
		'max_features': [2, 3, 4],
		'min_samples_leaf': [1, 2, 3],
		'min_samples_split': [4, 5, 6],
		'n_estimators': [400, 600, 800, 1000]
	}

	#grid_search_best_estimator = GetBestParamsGridSearch(param_grid, X, y)
	"""
	{'bootstrap': False,
	 'max_depth': 10,
	 'max_features': 2,
	 'min_samples_leaf': 1,
	 'min_samples_split': 4,
	 'n_estimators': 600}
	"""
	# create RANDOM FOREST WITH THE BEST VERSION OF PARAMETERS FOR FRONT BACK
	gridSearchRandomForest = RandomForestClassifier(n_estimators=600, bootstrap=False, max_depth=10, max_features=2,
									min_samples_leaf=1, min_samples_split=4)
	gridSearchRandomForest = FitRF(X, y, gridSearchRandomForest)
	CreateConfusionMatrix('frontBack', dataForCM, gridSearchRandomForest, "GridSearchForestMatrixFrontBack", plt.cm.Reds)

	# ########################################## ANGULAR ##################################
	# estimate some value of hyperparameters
	X, y = GetLabelAndFeatureData('angular', dataForRF)
	# bestEstimatorRandom = GetBestParamsRandomSearch(X, y)
	"""
	BEST PARAMS
	{'bootstrap': False,
	 'max_depth': 30,
	 'max_features': 'sqrt',
	 'min_samples_leaf': 4,
	 'min_samples_split': 5,
	 'n_estimators': 200}
	"""
	RandomRandomForest = RandomForestClassifier(n_estimators=200, bootstrap=False, max_depth=30, max_features='sqrt',
												min_samples_leaf=4, min_samples_split=5)
	RandomRandomForest = FitRF(X, y, RandomRandomForest)
	CreateConfusionMatrix('angular', dataForCM, RandomRandomForest, "BestEstimatorRandomA", plt.cm.Reds)

	# create random forest with default parameters for comparison
	basicForest = RandomForestClassifier()
	basicForest = FitRF(X, y, basicForest)
	CreateConfusionMatrix('angular', dataForCM, basicForest, "BasicForestMatrixA", plt.cm.Reds)

	# param grid based on params estimated by random search
	param_grid = {
		'bootstrap': [False],
		'max_depth': [20, 30, 40],
		'max_features': [2, 3, 4],
		'min_samples_leaf': [3, 4, 5],
		'min_samples_split': [4, 5, 6],
		'n_estimators': [100, 200, 400]
	}

	# grid_search_best_estimator = GetBestParamsGridSearch(param_grid, X, y)
	"""
	{'bootstrap': False,
	 'max_depth': 20,
	 'max_features': 2,
	 'min_samples_leaf': 4,
	 'min_samples_split': 4,
	 'n_estimators': 100}
	"""
	# create RANDOM FOREST WITH THE BEST VERSION OF PARAMETERS FOR FRONT BACK
	gridSearchRandomForest = RandomForestClassifier(n_estimators=100, bootstrap=False, max_depth=20, max_features=2,
													min_samples_leaf=4, min_samples_split=4)
	gridSearchRandomForest = FitRF(X, y, gridSearchRandomForest)
	CreateConfusionMatrix('angular', dataForCM, gridSearchRandomForest, "GridSearchForestMatrixAngular",
						  plt.cm.Reds)


