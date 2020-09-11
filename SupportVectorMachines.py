import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pprint import pprint

from sklearn import svm, datasets

from CreateDataMatrix import dataDTSecondSet as dataForSVM
from CreateDataMatrix import dataForDTRealImagFrozenDict as dataForSVMOlder
from CreateDataMatrix import dataDTSecondCM as dataForCMSecond
from ConfusionMatrix import CreateConfusionMatrix
from CreateDataMatrix import dataForCM
from BuildDT import GetLabel
from BuildDT import GetFeatures

# this modul tunes hyperparameters for SVM and subsequently creates SVM classifier with the best hyperparamaters found

def TuneParamsForSVM(labelColoumnName, dataForSVM):
	"""
	tunes hyperparameters for support vector machine using 3,4 and 5 cross validation
	:param labelColoumnName:
	:param dataForSVM:
	:return: SVM classifier with the best hyperparameter combination found
	"""
	# get labels and features
	labels = GetLabel(labelColoumnName, dataForSVM)
	features = GetFeatures(dataForSVM)
	# create basic SVM classifier
	svm_classifier = svm.SVC()
	# list of parameters to try by cross validation
	kernels = ['linear', 'rbf', 'poly']
	params_to_try = [{'C': [1, 10, 100], 'kernel': kernels, 'gamma': [1, 10, 30, 50, 70, 100]}]
	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20,
														stratify=labels, random_state=42)
	# data standardization
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	y_train = np.ravel(y_train)
	x_test = scaler.transform(x_test)

	# cross validation
	score = 0
	for cv in range(3, 6):
		create_grid = GridSearchCV(svm_classifier, param_grid=params_to_try, cv=cv)
		create_grid.fit(x_train, y_train)
		newScore = create_grid.score(x_test, y_test)
		print("score for %d fold cross validation is %3.2f" % (cv, newScore))
		print("best fit params:")
		currentBestParams = create_grid.best_params_
		print(currentBestParams)
		if newScore > score:
			score = newScore
			bestParams = currentBestParams
			bestEstimator = create_grid.best_estimator_
	return bestEstimator, scaler, x_test, y_test


if __name__ == "__main__":
	# SVM for newer data
	leftRightEstimator, leftRightScaler, x_test, y_test = TuneParamsForSVM('leftRight', dataForSVM)
	CreateConfusionMatrix('leftRight', dataForCM, leftRightEstimator, 'leftRightSVM', plt.cm.Reds, leftRightScaler)
	newScore = leftRightEstimator.score(x_test, y_test)
	print("score for is %3.2f" % newScore)

	frontBackEstimator, frontBackScaler, x_test, y_test= TuneParamsForSVM('frontBack', dataForSVM)
	CreateConfusionMatrix('frontBack', dataForCM, frontBackEstimator, 'frontBackSVM', plt.cm.Reds, frontBackScaler)

	angularEstimator, angularScaler,x_test, y_test = TuneParamsForSVM('angular', dataForSVM)
	CreateConfusionMatrix('angular', dataForCM, angularEstimator, 'angularSVM', plt.cm.Reds, angularScaler)

	# SVM for older data
	leftRightEstimator, leftRightScaler,x_test, y_test = TuneParamsForSVM('leftRight', dataForSVMOlder)
	CreateConfusionMatrix('leftRight', dataForCMSecond, leftRightEstimator, 'leftRightSVMOlderData', plt.cm.Reds, leftRightScaler)

	frontBackEstimator, frontBackScaler,x_test, y_test = TuneParamsForSVM('frontBack', dataForSVMOlder)
	CreateConfusionMatrix('frontBack', dataForCMSecond, frontBackEstimator, 'frontBackSVMOlderData', plt.cm.Reds, frontBackScaler)

	angularEstimator, angularScaler,x_test, y_test = TuneParamsForSVM('angular', dataForSVMOlder)
	CreateConfusionMatrix('angular', dataForCMSecond, angularEstimator, 'angularSVMOlderData', plt.cm.Reds, angularScaler)



