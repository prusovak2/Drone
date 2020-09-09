from BuildDT import BuildDT
from ConfusionMatrix import CreateConfusionMatrix
from BuildDT import GetFeatures, GetLabel
from CreateDataMatrixForDT import dataForDTRealImagFrozenDict as dataForModel
from BuildRandomForest import GetLabelAndFeatureData
from CreateDataMatrixForDT import dataForCM
from SupportVectorMachines import TuneParamsForSVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def ScoreModel(labelColumnName, model, dataForCM):
	X,y = GetLabelAndFeatureData(labelColumnName, dataForCM)
	score = model.score(X, y)
	return score

def ScoreModelScaledFeatures(labelColumnName, model, dataForCM, scaledFeatures):
	y = GetLabel(labelColumnName,dataForCM)
	score = model.score(scaledFeatures, y)
	return score

def EvaluateModels(labelColumnName, dataForModel, dataForCM, CMColor, specification, outputFile):
	decisionTree, x_train, y_train, x_test, y_test = BuildDT(labelColumnName, dataForModel)
	CreateConfusionMatrix(labelColumnName, dataForCM, decisionTree, "decisionTreeCM "+ labelColumnName, CMColor)
	DTscore = ScoreModel(labelColumnName,decisionTree,dataForCM)
	print(specification)
	print('decision tree score is %3.2f' % DTscore)

	svm, scaler, x_test, y_test = TuneParamsForSVM(labelColumnName, dataForModel)
	transformedFeatures = CreateConfusionMatrix(labelColumnName, dataForCM, svm, 'SVM_CM '+ labelColumnName, CMColor, scaler)
	SVMScore = ScoreModelScaledFeatures(labelColumnName,svm,dataForCM,transformedFeatures)
	print(specification, labelColumnName)
	print('SVM score is %3.2f' % SVMScore)

	with open(outputFile,mode='a+') as output:
		output.write(specification + labelColumnName +'\n')
		output.write('decision tree score is %3.2f\n' % DTscore)
		output.write('SVM score is %3.2f\n' % SVMScore)
	return DTscore, SVMScore


labels = ['leftRight', 'frontBack', 'angular']
for label in labels:
	EvaluateModels(label, dataForModel, dataForCM, plt.cm.Blues, "IntervalLen: 40, representant sample 0 ",'OutputStages\\scores.txt')

