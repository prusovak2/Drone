from differentPreprocessing import PrepareData
from pprint import pprint
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from CreateDataMatrix import MakeCMDsDiscreteWithFrozenDict, frozenCmds
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from differentPreprocessing import GetLabel,GetFeatures
from BuildDT import BuildDT
from ConfusionMatrix import CreateConfusionMatrix
from SupportVectorMachines import TuneParamsForSVM

class Dataset:
	def __init__(self, navdataFileName, cmdsFileName, datasetName):
		self.NavdataFile = navdataFileName
		self.CommandsFile = cmdsFileName
		self.DatasetName = datasetName

Datasets = []
basic = Dataset('InputData\\navdata.tsv', 'InputData\\commands.tsv', 'basic')
Datasets.append(basic)
basicCM = Dataset('InputData\\navdataCMTABS.tsv', 'InputData\\commandsCM.tsv', 'basicCM')
Datasets.append(basicCM)
secondSet = Dataset('InputData\\navdataSecondSetTABS.tsv', 'InputData\\cmdsSecondSet.tsv', 'secondSet')
Datasets.append(secondSet)
secondSetCM = Dataset('InputData\\navdataSecondCMTABS.tsv', 'InputData\\cmdsSecondCM.tsv', 'secondCM')
Datasets.append(secondSetCM)
leftRight = Dataset('InputData\\leftRightNavdataTABS.tsv', 'InputData\\leftRightCMDS.tsv', 'leftRight')
Datasets.append(leftRight)

fileNameStart = 'InputData\\myOwndata\\'
filenameEndCMDs = '\\commands.tsv'
filenameEndNAVDATA = '\\navdataTABS.tsv'
files = ['angular', 'drone_data', 'drone_data3', 'frontBack', 'sada2', 'straight', 'straight4', 'straightAllSides', 'straightAllSidesBetter']

testDatasets = []
for file in files:
	if file != 'angular':
		testDatasets.append(Dataset(fileNameStart+file+filenameEndNAVDATA, fileNameStart+file+filenameEndCMDs, file))

Datasets.append(Dataset(fileNameStart+'angular'+filenameEndNAVDATA, fileNameStart+file+filenameEndCMDs, file))


DataMatrices = []
for dataset in Datasets:
	matrix = PrepareData(dataset.CommandsFile, dataset.NavdataFile)
	DataMatrices.append(matrix)

TestDataMatrices = []
for dataset in testDatasets:
	matrix = PrepareData(dataset.CommandsFile, dataset.NavdataFile)
	TestDataMatrices.append(matrix)

trainMatrix = pd.concat(DataMatrices)
testMatrix = pd.concat(TestDataMatrices)
trainMatrix.reset_index(drop=True, inplace=True)
testMatrix.reset_index(drop=True, inplace=True)

trainMatrix.to_csv('OutputStages\\trainMatrix.tsv', sep='\t')
testMatrix.to_csv('OutputStages\\testMatrix.tsv', sep='\t')

print(len(DataMatrices))
pprint(DataMatrices)

labels = ['leftRight', 'frontBack', 'angular']
for label in labels:
	dt,_,_,_,_ = BuildDT(label, trainMatrix, False)
	CreateConfusionMatrix(label,testMatrix,dt,'DT '+label, plt.cm.Blues, time=False)
	svm, scaler,_,_ = TuneParamsForSVM(label, trainMatrix, False)
	CreateConfusionMatrix(label, testMatrix, svm, 'SVM ' + label, plt.cm.Blues, time=False, scaler=scaler)

"""
decisionTree = DecisionTreeClassifier(criterion='gini', max_depth=4, class_weight='balanced')
features_train = GetFeatures(trainMatrix)
labels_train = GetLabel('leftRight', trainMatrix)

decisionTree.fit(features_train, labels_train)

features_test = GetFeatures(testMatrix)
labels_test = GetLabel('leftRight', testMatrix)
disp = plot_confusion_matrix(decisionTree, features_test, labels_test,
							 display_labels=[*frozenCmds.values()],
							 cmap=plt.cm.Blues,
							 normalize=None)
disp.ax_.set_title("leftRight")
print('leftRight')
print(disp.confusion_matrix)
plt.show()
plt.close()

features_train = GetFeatures(trainMatrix)
labels_train = GetLabel('frontBack', trainMatrix)

decisionTree.fit(features_train, labels_train)

features_test = GetFeatures(testMatrix)
labels_test = GetLabel('frontBack', testMatrix)
disp = plot_confusion_matrix(decisionTree, features_test, labels_test,
							 display_labels=[*frozenCmds.values()],
							 cmap=plt.cm.Blues,
							 normalize=None)
disp.ax_.set_title("frontBack")
print('frontBack')
print(disp.confusion_matrix)
plt.show()
plt.close()

features_train = GetFeatures(trainMatrix)
labels_train = GetLabel('angular', trainMatrix)

decisionTree.fit(features_train, labels_train)

features_test = GetFeatures(testMatrix)
labels_test = GetLabel('angular', testMatrix)
disp = plot_confusion_matrix(decisionTree, features_test, labels_test,
							 display_labels=[*frozenCmds.values()],
							 cmap=plt.cm.Blues,
							 normalize=None)
disp.ax_.set_title("angular")
print('angular')
print(disp.confusion_matrix)
plt.show()
plt.close()
"""

