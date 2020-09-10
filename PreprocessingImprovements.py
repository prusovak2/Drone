from BuildDT import BuildDT
from ConfusionMatrix import CreateConfusionMatrix
from BuildDT import GetFeatures, GetLabel
from CreateDataMatrixForDT import dataForDTRealImagFrozenDict as dataForModel
from BuildRandomForest import GetLabelAndFeatureData
from CreateDataMatrixForDT import dataForCM
from CreateDataMatrixForDT import CreateDataFrameForDTMatrixShift, merged
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


def EvaluateModels(labelColumnName, dataForModel, dataForCM, CMColor, specification, outputFile, showCM=True ):
	decisionTree, x_train, y_train, x_test, y_test = BuildDT(labelColumnName, dataForModel)
	CreateConfusionMatrix(labelColumnName, dataForCM, decisionTree, "decisionTreeCM " + labelColumnName, CMColor,showCM=showCM)
	DTscore = ScoreModel(labelColumnName,decisionTree,dataForCM)
	print(specification)
	print('decision tree score is %3.2f' % DTscore)

	svm, scaler, x_test, y_test = TuneParamsForSVM(labelColumnName, dataForModel)
	transformedFeatures = CreateConfusionMatrix(labelColumnName, dataForCM, svm, 'SVM_CM '+ labelColumnName, CMColor, scaler, showCM)
	SVMScore = ScoreModelScaledFeatures(labelColumnName,svm,dataForCM,transformedFeatures)
	print(specification, labelColumnName)
	print('SVM score is %3.2f' % SVMScore)

	outputFile.write(specification + labelColumnName +'\n')
	outputFile.write('decision tree score is %3.2f\n' % DTscore)
	outputFile.write('SVM score is %3.2f\n' % SVMScore)
	return DTscore, SVMScore


def deleteContent(fName):
	with open(fName, "w"):
		pass


def TryShiftsForOneLabel(labelColumnName, mergedData,outputFileName, intervalLen=40, lowerShiftBorder=0, upperShiftBorder=20):
	with open(outputFileName,"w") as outputFile:
		bestDTscore = 0
		bestSVMscore = 0
		for shift in range(lowerShiftBorder, upperShiftBorder + 1):
			shifted = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
														representantSampleShift=shift)
			DTscore, SVMscore = EvaluateModels(labelColumnName, shifted, dataForCM, plt.cm.Blues,
													"IntervalLen: " + str(intervalLen) + " representant sample " + str(
													shift) + " ", outputFile, showCM=False)
			if DTscore > bestDTscore:
				bestDTscore = DTscore
				bestDTShift = shift
			if SVMscore > bestSVMscore:
				bestSVMscore = SVMscore
				bestSVMshift = shift
		print('bestDTscore' + str(bestDTscore))
		print('bestDTshift' + str(bestDTShift))
		print('bestSVMscore' + str(bestSVMscore))
		print('bestSVMshift' + str(bestSVMshift))
		outputFile.write("\n")
		outputFile.write('bestDTscore' + str(bestDTscore) + "\n")
		outputFile.write('bestDTshift' + str(bestDTShift) + "\n")
		outputFile.write('bestSVMscore' + str(bestSVMscore) + "\n")
		outputFile.write('bestSVMshift' + str(bestSVMshift) + "\n")




labels = ['leftRight', 'frontBack', 'angular']
# TryShiftsForOneLabel('leftRight', merged, 'OutputStages\\scoresShiftLeftRight.txt')
# TryShiftsForOneLabel('frontBack', merged, 'OutputStages\\scoresShiftFrontBack.txt')
TryShiftsForOneLabel('angular', merged, 'OutputStages\\scoresShiftAngular.txt')