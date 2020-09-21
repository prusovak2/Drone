from collections import Counter as Count
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from CreateDataMatrix import MakeCMDsDiscreteWithFrozenDict, frozenCmds
from sklearn.model_selection import train_test_split
import sklearn.tree
import random

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
#from CreateDataMatrix import frozenCmds
from pprint import pprint

class TimeFrame:
    """
    Class to store 250 ms interval of commands and navdata during preprocessing
    before functions (mean,std, fft) are used and result is stored in pandas DataFrame
    """
    def __init__(self):
        """
        all cmds in particular 250 ms long interval
        """
        self.commands=[]
        """
        all navdata in particular 250 ms long interval
        """
        self.navdata=[]
        """
        begining of an interval
        """
        self.initial_time=None
        """
        most common cmd in interval represents interval
        """
        self.most_common_commands=None
        """
        only roll, pitch and yaw columns selected from navdata
        """
        self.selected_navdata=None


def select_label(listOfLabelVals):
    """
    provided values of label from particular interval, returns the most common label value in the interval
    :param listOfLabelVals:
    :return:
    """
    return sorted(Count(listOfLabelVals).items(), key=lambda x: x[1])[-1][0]


def ReadData(cmdFileName, navdataFileName):
    """
    reads command and navdata files into 2D arrays
    :param cmdFileName: file containing commands
    :param navdataFileName: file containing navdata
    :return: commands array, navdata array
    """
    navdata = []
    with open(navdataFileName, "r") as f:
        for line in f:
            if (line[-1] == "\n"):
                line = line[:-1]
            data = line.split("\t")
            data = list(map(float, data[1:]))
            navdata.append(data)
    navdata = sorted(navdata, key=lambda x: x[0])

    commands = []
    with open(cmdFileName, "r") as f:
        for line in f:
            if (line[-1] == "\n"):
                line = line[:-1]
            data = line.split("\t")
            commands.append([float(data[0])] + data[1:])
    commands = sorted(commands, key=lambda x: x[0])
    return commands, navdata


def PrepareData(cmdFileName, navdataFileName):
    """
    reads commands and navdata, divides them into 250 ms long intervals,
    picks the most common cmd from interval to represent interval,
    counts mean, std, mean and std of fft of roll, pitch and yaw navdata columns within intervals
    :param cmdFileName: file to read commands from
    :param navdataFileName: file to read navdata from
    :return: pandas DataFrame with one row for each 250 ms long interval
    """
    commands, navdata = ReadData(cmdFileName, navdataFileName)

    # skip all navdata that were measured before the first cmd
    labelled = []
    i = 0
    j = 0
    while i < len(navdata) and navdata[i][0] < commands[0][0]:
        i += 1

    # divide navdata and cmds into 250 ms long intervals
    while True:
        tf = TimeFrame()
        # begining af the interval - time of the first cmd in the interval
        tf.initial_time = commands[j][0]
        tf.commands.append(commands[j])
        labelled.append(tf)  # list of timeFrames - of all intervals
        j += 1
        # take all cmds that occured within 250 ms since the first cmd in the interval
        while j < len(commands) and commands[j][0] < labelled[-1].initial_time + 250:
            labelled[-1].commands.append(commands[j])
            j += 1

        # take all navdata that occured within 250 ms since the first cmd in the interval
        while i < len(navdata) and navdata[i][0] < labelled[-1].initial_time + 250:
            labelled[-1].navdata.append(navdata[i])
            i += 1

        if (j == len(commands) or i == len(navdata)):
            break

    i = 0
    while i < len(labelled):
        # label of the interval is to be the most common command in the interval
        labels = [
            select_label(list(map(lambda x: x[1], labelled[i].commands))),  # leftRight
            select_label(list(map(lambda x: x[2], labelled[i].commands))),  # frontBack
            # select_label(list(map(lambda x: x[3], labelled[i].commands))), # up - not used
            select_label(list(map(lambda x: x[4], labelled[i].commands))),  # angular
        ]
        labelled[i].most_common_commands = labels

        # pick roll, pitch and yaw from navdata
        data = np.array(labelled[i].navdata)[:, 12:15]
        labelled[i].selected_navdata = data
        i += 1

    # insert data into pandas DataFrame
    dataColumnNamesRealImag = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean_Real',
                           'Roll_FFT_Mean_Imag',
                           'Roll_FFT_SD', 'Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean_Real', 'Pitch_FFT_Mean_Imag',
                           'Pitch_FFT_SD',
                           'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean_Real', 'Yaw_FFT_Mean_Imag', 'Yaw_FFT_SD']

    # determine the shape of DataFrame
    numRecords = len(labelled)
    ind = list()
    for k in range(0, numRecords):
        ind.append(k)
    print(numRecords)
    pprint(ind)
    dataMatrix = pd.DataFrame(columns=dataColumnNamesRealImag, index=ind)

    # to make a code more readable
    navdataTranslate = {'Pitch': 0, 'Roll': 1, 'Yaw': 2}
    cmdTranslate = {'leftRight': 0, 'frontBack': 1, 'angular': 2}

    # fill DataFrame with data, use mean, std, mean and std of fft on roll, pitch, yaw columns
    index = 0
    for timeFrame in labelled:
        # labels
        dataMatrix.leftRight[index] = float(timeFrame.most_common_commands[cmdTranslate['leftRight']])
        dataMatrix.frontBack[index] = float(timeFrame.most_common_commands[cmdTranslate['frontBack']])
        dataMatrix.angular[index] = float(timeFrame.most_common_commands[cmdTranslate['angular']])

        # means
        means = timeFrame.selected_navdata.mean(axis=0)
        dataMatrix.Roll_Mean[index] = means[navdataTranslate['Roll']]
        dataMatrix.Pitch_Mean[index] = means[navdataTranslate['Pitch']]
        dataMatrix.Yaw_Mean[index] = means[navdataTranslate['Yaw']]

        # stds
        stds = timeFrame.selected_navdata.std(axis=0)
        dataMatrix.Roll_SD[index] = stds[navdataTranslate['Roll']]
        dataMatrix.Pitch_SD[index] = stds[navdataTranslate['Pitch']]
        dataMatrix.Yaw_SD[index] = stds[navdataTranslate['Yaw']]

        # roll FFT
        rollFFT = np.fft.fft(timeFrame.selected_navdata[:, navdataTranslate['Roll']])
        rollFFtmean = np.mean(rollFFT)
        dataMatrix.Roll_FFT_Mean_Real[index] = rollFFtmean.real
        dataMatrix.Roll_FFT_Mean_Imag[index] = rollFFtmean.imag
        rollFFTstd = np.std(rollFFT)
        dataMatrix.Roll_FFT_SD[index] = rollFFTstd

        # pitch FFT
        pitchFFT = np.fft.fft(timeFrame.selected_navdata[:, navdataTranslate['Pitch']])
        pitchFFtmean = np.mean(pitchFFT)
        dataMatrix.Pitch_FFT_Mean_Real[index] = pitchFFtmean.real
        dataMatrix.Pitch_FFT_Mean_Imag[index] = pitchFFtmean.imag
        pitchFFTstd = np.std(pitchFFT)
        dataMatrix.Pitch_FFT_SD[index] = pitchFFTstd

        # yaw FFT
        yawFFT = np.fft.fft(timeFrame.selected_navdata[:, navdataTranslate['Yaw']])
        yawFFtmean = np.mean(yawFFT)
        dataMatrix.Yaw_FFT_Mean_Real[index] = yawFFtmean.real
        dataMatrix.Yaw_FFT_Mean_Imag[index] = yawFFtmean.imag
        yawFFTstd = np.std(yawFFT)
        dataMatrix.Yaw_FFT_SD[index] = yawFFTstd

        index += 1

    # make commands discrete
    newDataMatrix = dataMatrix
    for i in ind:
        MakeCMDsDiscreteWithFrozenDict(i, dataMatrix, newDataMatrix)

    return newDataMatrix

def GetLabel(labelColumn, dataMatrix):
    '''
    separates a column to be predicted
    :param labelColumn:
    :param dataMatrix:
    :return: selected label column
    '''
    labels = dataMatrix[[labelColumn]]
    return labels

def GetFeatures(dataMatrix):
    '''
    separates columns to base a prediction on
    :param dataMatrix:
    :return: feature columns
    '''
    features = dataMatrix.drop(['leftRight', 'frontBack', 'angular'], axis=1)
    return features

dataMatrix = PrepareData('InputData\\commands.tsv', 'InputData\\navdata.tsv')
validationData = PrepareData('InputData\\commandsCM.tsv', 'InputData\\navdataCMTABS.tsv')
dataMatrix.to_csv('OutputStages\\differentApproach.tsv', sep='\t')
validationData.to_csv('OutputStages\\differentApproachCM.tsv', sep='\t')

# build DT
decisionTree = DecisionTreeClassifier(criterion='gini', max_depth=4, class_weight='balanced')
features_train = GetFeatures(validationData)
labels_train = GetLabel('leftRight', validationData)

decisionTree.fit(features_train, labels_train)

features_test = GetFeatures(dataMatrix)
labels_test = GetLabel('leftRight', dataMatrix)
disp = plot_confusion_matrix(decisionTree, features_test, labels_test,
                             display_labels=[*frozenCmds.values()],
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title("leftRight")
print('leftRight')
print(disp.confusion_matrix)
plt.show()
plt.close()

features_train = GetFeatures(validationData)
labels_train = GetLabel('frontBack', validationData)

decisionTree.fit(features_train, labels_train)

features_test = GetFeatures(dataMatrix)
labels_test = GetLabel('frontBack', dataMatrix)
disp = plot_confusion_matrix(decisionTree, features_test, labels_test,
                             display_labels=[*frozenCmds.values()],
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title("frontBack")
print('frontBack')
print(disp.confusion_matrix)
plt.show()
plt.close()

features_train = GetFeatures(validationData)
labels_train = GetLabel('angular', validationData)

decisionTree.fit(features_train, labels_train)

features_test = GetFeatures(dataMatrix)
labels_test = GetLabel('angular', dataMatrix)
disp = plot_confusion_matrix(decisionTree, features_test, labels_test,
                             display_labels=[*frozenCmds.values()],
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title("angular")
print('angular')
print(disp.confusion_matrix)
plt.show()
plt.close()
