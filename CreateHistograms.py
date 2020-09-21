import pandas as pd
from collections import Counter
from pprint import pprint
import matplotlib.pyplot as plt

def CreateHistogram(CMDfilename):
	"""
	for given command file creates histogram showing frequency of each command
	:param CMDfilename:
	:return:
	"""
	cmdsColumnNames = ['time', 'leftRight', 'frontBack', 'up', 'angular']
	cmds = pd.read_csv(CMDfilename, delimiter='\t', header=None, names=cmdsColumnNames)

	lr = cmds.leftRight
	convert_label = {-0.25: '-', 0.0: '0', 0.25: '+'}
	lr = list(map(lambda x: convert_label[x], lr))
	lr = list(map(lambda x: x+' LR', lr))

	fb = cmds.frontBack
	fb = list(map(lambda x: convert_label[x], fb))
	fb = list(map(lambda x: x+' FB', fb))

	a = cmds.angular
	pprint(a)
	a = list(map(lambda x: convert_label[x], a))
	a = list(map(lambda x: x + ' A', a))
	pprint(a)

	print("ABRAKA")
	toCount = lr + fb + a
	counts = Counter(toCount)
	df = pd.DataFrame.from_dict(counts, orient='index')
	df.plot(kind='bar', title=CMDfilename+' histogram')
	plt.show()


if __name__ == "__main__":
	files = [
		'InputData\\commands.tsv',
		'InputData\\commandsCM.tsv',
		'InputData\\cmdsSecondSet.tsv',
		'InputData\\cmdsSecondCM.tsv',
		'InputData\\leftRightCMDS.tsv',
	]

	fileNameStart = 'InputData\\myOwndata\\'
	filenameEndCMDs = '\\commands.tsv'
	names = ['angular', 'drone_data', 'drone_data3', 'frontBack', 'sada2', 'straight', 'straight4', 'straightAllSides', 'straightAllSidesBetter']

	for name in names:
		fileName = fileNameStart+name+filenameEndCMDs
		files.append(fileName)

	for fileName in files:
		CreateHistogram(fileName)