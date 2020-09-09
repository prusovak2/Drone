import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pprint import pprint

from sklearn import svm, datasets

from CreateDataMatrixForDT import dataDTSecondSet as dataForSVM
from ConfusionMatrix import CreateConfusionMatrix
from CreateDataMatrixForDT import dataForCM
from BuildDT import GetLabel
from BuildDT import GetFeatures


def plotSVC(title,features,labels, svc):
	y = labels[:]
	X = features.values
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	h = (x_max / x_min)/100
	print(x_min, x_max, h, y_min, y_max)
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	plt.subplot(1, 1, 1)
	Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.title(title)
	plt.show()


labels = GetLabel('leftRight', dataForSVM)
features = GetFeatures(dataForSVM)
svm_classifier = svm.SVC()
# pipe_steps = [('scaler', scaler), ('svm_classifier', svm_classifier)]
# pipeline = Pipeline(pipe_steps)
kernels = ['linear', 'rbf', 'poly']
params_to_try = [{'C': [1, 10, 100], 'kernel': kernels, 'gamma': [1, 10, 30, 50,70, 100]}]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20,
													stratify=labels, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = np.ravel(y_train)

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
		bestEstimator =create_grid.best_estimator_

CreateConfusionMatrix('leftRight', dataForCM, bestEstimator, 'firstSVMCM', plt.cm.Reds, scaler)

baseSVM = svm.SVC()
baseSVM.fit(x_train, y_train)
CreateConfusionMatrix('leftRight', dataForCM, baseSVM, 'baseSVM', plt.cm.Reds, scaler)

"""
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. We could
# avoid this ugly slicing by using a two-dim dataset
y = iris.target
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
	svc = svm.SVC(kernel=kernel).fit(X, y)
	plotSVC('kernel=' + str(kernel))
"""



