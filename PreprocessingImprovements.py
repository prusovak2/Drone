from BuildDT import BuildDT
from ConfusionMatrix import CreateConfusionMatrix
from BuildDT import GetFeatures, GetLabel
from CreateDataMatrixForDT import dataForDTRealImagFrozenDict as dataForModel
from BuildRandomForest import GetLabelAndFeatureData
from CreateDataMatrixForDT import dataForCM
from CreateDataMatrixForDT import CreateDataFrameForDTMatrixShift, merged
from SupportVectorMachines import TuneParamsForSVM
import matplotlib.pyplot as plt
from pprint import pprint
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
	with open(outputFileName, "w") as outputFile:
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
		return bestDTscore, bestDTShift, bestSVMscore, bestSVMshift


def TryShiftsForAllLabels(mergedData, outputFileName, intervalLen=40, lowerShiftBorder=0, upperShiftBorder=20):
	with open(outputFileName, "w") as outputFile:
		bestDTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestSVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestDTshift = {'leftRight': None, 'frontBack': None, 'angular': None}
		bestSVMshift = {'leftRight': None, 'frontBack': None, 'angular': None}
		labels = ['leftRight', 'frontBack', 'angular']
		bestDTscoreAverageLabels = 0
		bestSVMscoreAverageLabels = 0
		for shift in range(lowerShiftBorder, upperShiftBorder + 1):
			shifted = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
													  representantSampleShift=shift)
			DTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			SVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			for label in labels:
				DTscore[label], SVMscore[label] = EvaluateModels(label, shifted, dataForCM, plt.cm.Blues,
														" IntervalLen: " + str(intervalLen) + " representant sample " + str(
														shift) + " ", outputFile, showCM=False)
				if DTscore[label] > bestDTscore[label]:
					bestDTscore[label] = DTscore[label]
					bestDTshift[label] = shift
				if SVMscore[label] > bestSVMscore[label]:
					bestSVMscore[label] = SVMscore[label]
					bestSVMshift[label] = shift
			averageDTscore = (DTscore['leftRight']+DTscore['frontBack'] + DTscore['angular'])/3
			averageSVMscore = (SVMscore['leftRight']+SVMscore['frontBack'] + SVMscore['angular'])/3
			outputFile.write('average DT score: ' + str(averageDTscore)+'\n')
			outputFile.write('average SVM score:' + str(averageSVMscore) + '\n')
			if averageDTscore > bestDTscoreAverageLabels:
				bestDTscoreAverageLabels =averageDTscore
				bestDTAverageShift = shift
			if averageSVMscore > bestSVMscoreAverageLabels:
				bestSVMscoreAverageLabels = averageSVMscore
				bestSVMaverageShift = shift

		print('best DT average Score' + str(bestDTscoreAverageLabels))
		print('best DT averages shift' + str(bestDTAverageShift))
		print('best SVM average score' + str(bestSVMscoreAverageLabels))
		print('best SVM average shift' + str(bestSVMaverageShift))
		outputFile.write("\n")
		outputFile.write('best DT average Score ' + str(bestDTscoreAverageLabels) + "\n")
		outputFile.write('best DT averages shift ' + str(bestDTAverageShift) + "\n")
		outputFile.write('best SVM average score ' + str(bestSVMscoreAverageLabels) + "\n")
		outputFile.write('best SVM average shift ' + str(bestSVMaverageShift) + "\n")

		print('bestDTscores:')
		pprint(bestDTscore)
		print('bestDTshifts:')
		pprint(bestDTshift)
		print('bestSVMscores:')
		pprint(bestSVMscore)
		print('bestSVMshifts:')
		pprint(bestDTshift)

		outputFile.write('bestDTscores:\n')
		pprint(bestDTscore, stream=outputFile)
		outputFile.write('bestDTshifts:\n')
		pprint(bestDTshift, stream=outputFile)
		outputFile.write('bestSVMscores:\n')
		pprint(bestSVMscore, stream=outputFile)
		outputFile.write('bestSVMshifts:\n')
		pprint(bestDTshift, stream=outputFile)


def TryIntervalLenghtsOneLabel(labelColumnName, mergedData,outputFileName, shift=0, lowerIntearvalLenBorder=20, upperIntervalLenBorder=40):
	with open(outputFileName, "w") as outputFile:
		bestDTscore = 0
		bestSVMscore = 0
		for intervalLen in range(lowerIntearvalLenBorder, upperIntervalLenBorder + 1):
			dataMatrix = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
														representantSampleShift=shift)
			DTscore, SVMscore = EvaluateModels(labelColumnName, dataMatrix, dataForCM, plt.cm.Blues,
													"IntervalLen: " + str(intervalLen) + " representant sample " + str(
													shift) + " ", outputFile, showCM=False)
			if DTscore > bestDTscore:
				bestDTscore = DTscore
				bestDTintervalLen = intervalLen
			if SVMscore > bestSVMscore:
				bestSVMscore = SVMscore
				bestSVMintervalLen = intervalLen
		print('bestDTscore' + str(bestDTscore))
		print('bestDT interval len' + str(bestDTintervalLen))
		print('bestSVMscore' + str(bestSVMscore))
		print('bestSVM interval len' + str(bestSVMintervalLen))
		outputFile.write("\n")
		outputFile.write('bestDTscore ' + str(bestDTscore) + "\n")
		outputFile.write('bestDTIntervalLen ' + str(bestDTintervalLen) + "\n")
		outputFile.write('bestSVMscore ' + str(bestSVMscore) + "\n")
		outputFile.write('bestSVMintervalLen ' + str(bestSVMintervalLen) + "\n")
		return bestDTscore, bestDTintervalLen, bestSVMscore, bestSVMintervalLen


def TryIntervalLensForAllLabels(mergedData, outputFileName, shift=0, lowerIntearvalLenBorder=20, upperIntervalLenBorder=40):
	with open(outputFileName, "w") as outputFile:
		bestDTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestSVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestDTintervalLen = {'leftRight': None, 'frontBack': None, 'angular': None}
		bestSVMintervalLen = {'leftRight': None, 'frontBack': None, 'angular': None}
		labels = ['leftRight', 'frontBack', 'angular']
		bestDTscoreAverageLabels = 0
		bestSVMscoreAverageLabels = 0
		for intervalLen in range(lowerIntearvalLenBorder, upperIntervalLenBorder + 1):
			dataMatrix = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
													  representantSampleShift=shift)
			DTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			SVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			for label in labels:
				DTscore[label], SVMscore[label] = EvaluateModels(label, dataMatrix, dataForCM, plt.cm.Blues,
														" IntervalLen: " + str(intervalLen) + " representant sample " + str(
														shift) + " ", outputFile, showCM=False)
				if DTscore[label] > bestDTscore[label]:
					bestDTscore[label] = DTscore[label]
					bestDTintervalLen[label] = intervalLen
				if SVMscore[label] > bestSVMscore[label]:
					bestSVMscore[label] = SVMscore[label]
					bestSVMintervalLen[label] = intervalLen
			averageDTscore = (DTscore['leftRight']+DTscore['frontBack'] + DTscore['angular'])/3
			averageSVMscore = (SVMscore['leftRight']+SVMscore['frontBack'] + SVMscore['angular'])/3
			outputFile.write('average DT score: ' + str(averageDTscore)+'\n')
			outputFile.write('average SVM score:' + str(averageSVMscore) + '\n')
			if averageDTscore > bestDTscoreAverageLabels:
				bestDTscoreAverageLabels = averageDTscore
				bestDTAverageIntervalLen = intervalLen
			if averageSVMscore > bestSVMscoreAverageLabels:
				bestSVMscoreAverageLabels = averageSVMscore
				bestSVMaverageIntervalLen = intervalLen

		print('best DT average Score' + str(bestDTscoreAverageLabels))
		print('best DT averages intervalLen' + str(bestDTAverageIntervalLen))
		print('best SVM average score' + str(bestSVMscoreAverageLabels))
		print('best SVM average IntervalLen' + str(bestSVMaverageIntervalLen))
		outputFile.write("\n")
		outputFile.write('best DT average Score ' + str(bestDTscoreAverageLabels) + "\n")
		outputFile.write('best DT averages IntervalLen ' + str(bestDTAverageIntervalLen) + "\n")
		outputFile.write('best SVM average score ' + str(bestSVMscoreAverageLabels) + "\n")
		outputFile.write('best SVM average IntervalLen ' + str(bestSVMaverageIntervalLen) + "\n")

		print('bestDTscores:')
		pprint(bestDTscore)
		print('bestDTintervalLen:')
		pprint(bestDTintervalLen)
		print('bestSVMscores:')
		pprint(bestSVMscore)
		print('bestSVMintervalLen:')
		pprint(bestDTintervalLen)

		outputFile.write('bestDTscores:\n')
		pprint(bestDTscore, stream=outputFile)
		outputFile.write('bestDTintervalLen:\n')
		pprint(bestDTintervalLen, stream=outputFile)
		outputFile.write('bestSVMscores:\n')
		pprint(bestSVMscore, stream=outputFile)
		outputFile.write('bestSVMintervalLen:\n')
		pprint(bestDTintervalLen, stream=outputFile)


if __name__ == "__main__":
	# TryShiftsForOneLabel('leftRight', merged, 'OutputStages\\scoresShiftLeftRight.txt')
	# TryShiftsForOneLabel('frontBack', merged, 'OutputStages\\scoresShiftFrontBack.txt')
	# TryShiftsForOneLabel('angular', merged, 'OutputStages\\scoresShiftAngular.txt')

	TryShiftsForAllLabels(merged, 'OutputStages\\scoresShiftAllLabels.txt')

	#TryIntervalLenghtsOneLabel('leftRight', merged, 'OutputStages\\scoresIntervalLenLeftRight.txt')
	TryIntervalLensForAllLabels(merged, 'OutputStages\\scoresIntervalLenghtsAllLabels.txt')