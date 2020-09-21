from differentPreprocessing import PrepareData
from pprint import pprint
import pandas as pd

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
	testDatasets.append(Dataset(fileNameStart+file+filenameEndNAVDATA, fileNameStart+file+filenameEndCMDs, file))


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
