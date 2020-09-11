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

def ScoreModel(labelColumnName, model, dataForCM):
	"""
	finds out, how well model performs while classifying dataForCM
	:param labelColumnName:
	:param model: model whose performance is to be evaluated
	:param dataForCM: data that model was not trained on
	:return: score
	"""
	X,y = GetLabelAndFeatureData(labelColumnName, dataForCM)
	score = model.score(X, y)
	return score

def ScoreModelScaledFeatures(labelColumnName, model, dataForCM, scaledFeatures):
	"""
	finds out, how well model performs while classifying dataForCM
	takes test features that have already been standardized by the same StandardScaler that was used to standardize training data
	:param labelColumnName:
	:param model: model whose performance is to be evaluated
	:param dataForCM: data that model was not trained on
	:param scaledFeatures: test features scaled by the StandardScaler used o standardize training data
	:return: score
	"""
	y = GetLabel(labelColumnName,dataForCM)
	score = model.score(scaledFeatures, y)
	return score


def EvaluateModels(labelColumnName, dataForModel, dataForCM, CMColor, specification, outputFile, showCM=True ):
	"""
	tunes params for DT and SVM on given data, builds DT and SVM with the best params found and evaluates their perforamce
	plots confusion matrix when showCM is True
	:param labelColumnName:
	:param dataForModel: train data
	:param dataForCM: test data
	:param CMColor: color of confusion matrix, plt.cm.SomeColor
	:param specification: is to be written to output file, preferably information regarding tested preprocessing approach
	:param outputFile: opened file to which the specification and model scores are to be appended
	:param showCM: should confusion matrices be shown?
	:return: score of decision tree, score of support vector machine
	"""
	# tune param and build decision tree
	decisionTree, x_train, y_train, x_test, y_test = BuildDT(labelColumnName, dataForModel)
	# crete confusion matrix for the decision tree
	CreateConfusionMatrix(labelColumnName, dataForCM, decisionTree, "decisionTreeCM " + labelColumnName, CMColor,showCM=showCM)
	# evaluate how well decision tree performs on test data
	DTscore = ScoreModel(labelColumnName, decisionTree, dataForCM)
	print(specification)
	print('decision tree score is %3.2f' % DTscore)

	# tune params and build SVM
	svm, scaler, x_test, y_test = TuneParamsForSVM(labelColumnName, dataForModel)
	# crate CM for SVM
	transformedFeatures = CreateConfusionMatrix(labelColumnName, dataForCM, svm, 'SVM_CM '+ labelColumnName, CMColor, scaler, showCM)
	# evaluate how well SVM performes
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


def TryShiftsForOneLabel(labelColumnName, mergedData, outputFileName, intervalLen=40, lowerShiftBorder=0, upperShiftBorder=20):
	"""
	tries to tune data preprocessing approach
	for a specified label column tries by which record from the interval should the interval be represented
	in order to create the most effective ML model
	chosen record provides time and label (commnad) for the interval (therefore for the record in dataMatrix ML is based on)
	uses decision tree and SVM
	:param labelColumnName:
	:param mergedData: output of ReadResampleMerge method
	:param outputFileName: name of file to log model scores
	:param intervalLen: number of samples considered an one interval in merged data
	:param lowerShiftBorder: lower border of an interval of sample to be tried as representative, value from [0, intervalLen]
	:param upperShiftBorder: lower border of an interval of sample to be tried as representative, value from [0, intervalLen]
	:return: score of the best DT, score of the best SVM, the most effective representative shift for DT, the most effective representative shift for SVM
	"""
	with open(outputFileName, "w") as outputFile:
		bestDTscore = 0
		bestSVMscore = 0
		for shift in range(lowerShiftBorder, upperShiftBorder + 1):
			# data preprocessing with respect to given representative shift
			shifted = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
														representantSampleShift=shift)
			# tune params for, create and evaluate DT and SVM
			DTscore, SVMscore = EvaluateModels(labelColumnName, shifted, dataForCM, plt.cm.Blues,
													"IntervalLen: " + str(intervalLen) + " representant sample " + str(
													shift) + " ", outputFile, showCM=False)
			# remember the best result
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
	"""
	tries to tune data preprocessing approach
	for all label columns tries by which record from the interval should the interval be represented
	in order to create the most effective ML model
	chosen record provides time and label (commnad) for the interval (therefore for the record in dataMatrix ML is based on)
	uses decision tree and SVM
	counts average score of each model and preprocessing approach over all label columns
	finds the best shift for given data taking into account the performance an all label columns at once
	:param mergedData: output of ReadResampleMerge method
	:param outputFileName: name of file to log model scores
	:param intervalLen: number of samples considered an one interval in merged data
	:param lowerShiftBorder: lowerShiftBorder: lower border of an interval of sample to be tried as representative, value from [0, intervalLen]
	:param upperShiftBorder:
	"""
	with open(outputFileName, "w") as outputFile:
		# dictionaries to remember the best results so far
		bestDTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestSVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestDTshift = {'leftRight': None, 'frontBack': None, 'angular': None}
		bestSVMshift = {'leftRight': None, 'frontBack': None, 'angular': None}
		labels = ['leftRight', 'frontBack', 'angular']
		# best average result over all labels
		bestDTscoreAverageLabels = 0
		bestSVMscoreAverageLabels = 0
		for shift in range(lowerShiftBorder, upperShiftBorder + 1):
			# data preprocessing
			shifted = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
													  representantSampleShift=shift)
			# dictionaries to remember performance of models on current version of dataMatrix
			DTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			SVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			for label in labels:
				# tune params for, create and evaluate DT and SVM
				DTscore[label], SVMscore[label] = EvaluateModels(label, shifted, dataForCM, plt.cm.Blues,
														" IntervalLen: " + str(intervalLen) + " representant sample " + str(
														shift) + " ", outputFile, showCM=False)
				# remember the best result so far
				if DTscore[label] > bestDTscore[label]:
					bestDTscore[label] = DTscore[label]
					bestDTshift[label] = shift
				if SVMscore[label] > bestSVMscore[label]:
					bestSVMscore[label] = SVMscore[label]
					bestSVMshift[label] = shift
			# calculate average result over all label columns
			averageDTscore = (DTscore['leftRight']+DTscore['frontBack'] + DTscore['angular'])/3
			averageSVMscore = (SVMscore['leftRight']+SVMscore['frontBack'] + SVMscore['angular'])/3
			outputFile.write('average DT score: ' + str(averageDTscore)+'\n')
			outputFile.write('average SVM score:' + str(averageSVMscore) + '\n')
			# remember the best average result
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
	"""
	tries to tune data preprocessing approach
	tries how long should be an inteval of megred data that produces one record in dataMatrix
	in order to create the most effective ML model
	uses decision tree and SVM
	tries lenghts from [lowerIntearvalLenBorder, upperIntearvalLenBorder]
	:param labelColumnName:
	:param mergedData: output of ReadResampleMerge method
	:param outputFileName: name of file to log model scores
	:param shift: which sample from the interval should represent the interval
	:param lowerIntearvalLenBorder: [lowerIntearvalLenBorder, upperIntearvalLenBorder]
	:param upperIntervalLenBorder:
	:return: score of the best DT, score of the best SVM, best intervalLen fot DT, best intervalLen for SVM
	"""
	with open(outputFileName, "w") as outputFile:
		bestDTscore = 0
		bestSVMscore = 0
		for intervalLen in range(lowerIntearvalLenBorder, upperIntervalLenBorder + 1):
			if shift > intervalLen:
				raise AttributeError
			# data preprocessing
			dataMatrix = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
														representantSampleShift=shift)
			# tune params for, create and evaluate DT and SVM
			DTscore, SVMscore = EvaluateModels(labelColumnName, dataMatrix, dataForCM, plt.cm.Blues,
													"IntervalLen: " + str(intervalLen) + " representant sample " + str(
													shift) + " ", outputFile, showCM=False)
			# remember the best result so far
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
	"""
	tries to tune data preprocessing approach
	tries how long should be an inteval of megred data that produces one record in dataMatrix
	in order to create the most effective ML model
	uses decision tree and SVM
	tries lenghts from [lowerIntearvalLenBorder, upperIntearvalLenBorder]
	counts average score of each model and preprocessing approach over all label columns
	finds the best intervalLen for given data taking into account the performance an all label columns at once
	:param mergedData: output of ReadResampleMerge method
	:param outputFileName: name of file to log model scores
	:param shift: which sample from the interval should represent the interval
	:param lowerIntearvalLenBorder:
	:param upperIntervalLenBorder:
	"""
	with open(outputFileName, "w") as outputFile:
		# dictionaries to remember the best results so far
		bestDTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestSVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
		bestDTintervalLen = {'leftRight': None, 'frontBack': None, 'angular': None}
		bestSVMintervalLen = {'leftRight': None, 'frontBack': None, 'angular': None}
		labels = ['leftRight', 'frontBack', 'angular']
		# best average result over all labels
		bestDTscoreAverageLabels = 0
		bestSVMscoreAverageLabels = 0
		for intervalLen in range(lowerIntearvalLenBorder, upperIntervalLenBorder + 1):
			# data preprocessing
			dataMatrix = CreateDataFrameForDTMatrixShift(inputDFmerged=mergedData, intervalLen=intervalLen,
													  representantSampleShift=shift)
			# dictionaries to remember performance of models on current version of dataMatrix
			DTscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			SVMscore = {'leftRight': 0, 'frontBack': 0, 'angular': 0}
			for label in labels:
				# tune params for, create and evaluate DT and SVM
				DTscore[label], SVMscore[label] = EvaluateModels(label, dataMatrix, dataForCM, plt.cm.Blues,
														" IntervalLen: " + str(intervalLen) + " representant sample " + str(
														shift) + " ", outputFile, showCM=False)
				# remember the best result so far
				if DTscore[label] > bestDTscore[label]:
					bestDTscore[label] = DTscore[label]
					bestDTintervalLen[label] = intervalLen
				if SVMscore[label] > bestSVMscore[label]:
					bestSVMscore[label] = SVMscore[label]
					bestSVMintervalLen[label] = intervalLen
			# calculate average result over all label columns
			averageDTscore = (DTscore['leftRight']+DTscore['frontBack'] + DTscore['angular'])/3
			averageSVMscore = (SVMscore['leftRight']+SVMscore['frontBack'] + SVMscore['angular'])/3
			outputFile.write('average DT score: ' + str(averageDTscore)+'\n')
			outputFile.write('average SVM score:' + str(averageSVMscore) + '\n')
			# remember the best average result
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