from collections import Counter as C
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
    def __init__(self):
        self.commands=[]
        self.navdata=[]
        self.initial_time=None
        self.most_common_commands=None
        self.selected_navdata=None


def select_label(s):
    return sorted(C(s).items(), key=lambda x: x[1])[-1][0]

def PrepareData(cmdFileName, navdataFileName):
    # 1] nactu vsechna navdata a vsechny commands.
    #    data vzata z InputData/
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

    # 2] preskocim vsechna navdata, ktera jsem nameril pred prvnim commandem
    labelled = []
    i = 0
    j = 0
    while i < len(navdata) and navdata[i][0] < commands[0][0]:
        i += 1

    # 3] rozdelim data do skupinek tak, ze vzdycky vezmu command, a s nim vsechny commandy, ktere jsou do 250 ms po nem,
    #    a pridam k tomu vsechna navdata, ktera jsem jeste nikam nepriradil, a byla sebrana do 250ms od toho prikazu
    # Taky to neni idealni zpusob, jak to namatchovat, ale whatever. Navic jsou nektere skupiny daleko, daleko vetsi nez 50 zaznamu.

    while True:
        tf = TimeFrame()
        tf.initial_time = commands[j][0]
        tf.commands.append(commands[j])
        labelled.append(tf)
        j += 1
        while j < len(commands) and commands[j][0] < labelled[-1].initial_time + 250:
            labelled[-1].commands.append(commands[j])
            j += 1

        while i < len(navdata) and navdata[i][0] < labelled[-1].initial_time + 250:
            labelled[-1].navdata.append(navdata[i])
            i += 1

        if (j == len(commands) or i == len(navdata)):
            break

    # 4] label kazde skupiny je nejcastejsi command v ni.
    #    jako vstupni data beru sloupecky 11,12,13 (nultý je až ten po timestampu, tedy state. baterka je sloupec 1, atd).
    #    Navdata agreguji tak, ze je v ramci skupiny vsechny zprumeruji.

    i = 0
    while i < len(labelled):
        labels = [
            select_label(list(map(lambda x: x[1], labelled[i].commands))),  # leftRight
            select_label(list(map(lambda x: x[2], labelled[i].commands))),  # frontBack
            # select_label(list(map(lambda x: x[3], labelled[i].commands))),
            select_label(list(map(lambda x: x[4], labelled[i].commands))),  # angular
        ]
        labelled[i].most_common_commands = labels

        data = np.array(labelled[i].navdata)[:, 12:15]
        # data=data.mean(axis=0)
        labelled[i].selected_navdata = data
        i += 1

        dataColumnNamesRealImag = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean_Real',
                               'Roll_FFT_Mean_Imag',
                               'Roll_FFT_SD', 'Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean_Real', 'Pitch_FFT_Mean_Imag',
                               'Pitch_FFT_SD',
                               'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean_Real', 'Yaw_FFT_Mean_Imag', 'Yaw_FFT_SD']


    numRecords = len(labelled)
    ind = list()
    for k in range(0, numRecords):
        ind.append(k)
    print(numRecords)
    pprint(ind)
    dataMatrix = pd.DataFrame(columns=dataColumnNamesRealImag, index=ind)

    navdataTranslate = {'Pitch': 0, 'Roll': 1, 'Yaw': 2}
    cmdTranslate = {'leftRight': 0, 'frontBack': 1, 'angular': 2}

    index = 0
    for timeFrame in labelled:
        dataMatrix.leftRight[index] = float(timeFrame.most_common_commands[cmdTranslate['leftRight']])
        dataMatrix.frontBack[index] = float(timeFrame.most_common_commands[cmdTranslate['frontBack']])
        dataMatrix.angular[index] = float(timeFrame.most_common_commands[cmdTranslate['angular']])

        means = timeFrame.selected_navdata.mean(axis=0)
        dataMatrix.Roll_Mean[index] = means[navdataTranslate['Roll']]
        dataMatrix.Pitch_Mean[index] = means[navdataTranslate['Pitch']]
        dataMatrix.Yaw_Mean[index] = means[navdataTranslate['Yaw']]

        stds = timeFrame.selected_navdata.std(axis=0)
        dataMatrix.Roll_SD[index] = stds[navdataTranslate['Roll']]
        dataMatrix.Pitch_SD[index] = stds[navdataTranslate['Pitch']]
        dataMatrix.Yaw_SD[index] = stds[navdataTranslate['Yaw']]
        
        rollFFT = np.fft.fft(timeFrame.selected_navdata[:, navdataTranslate['Roll']])
        rollFFtmean = np.mean(rollFFT)
        dataMatrix.Roll_FFT_Mean_Real[index] = rollFFtmean.real
        dataMatrix.Roll_FFT_Mean_Imag[index] = rollFFtmean.imag
        rollFFTstd = np.std(rollFFT)
        dataMatrix.Roll_FFT_SD[index] = rollFFTstd

        pitchFFT = np.fft.fft(timeFrame.selected_navdata[:, navdataTranslate['Pitch']])
        pitchFFtmean = np.mean(pitchFFT)
        dataMatrix.Pitch_FFT_Mean_Real[index] = pitchFFtmean.real
        dataMatrix.Pitch_FFT_Mean_Imag[index] = pitchFFtmean.imag
        pitchFFTstd = np.std(pitchFFT)
        dataMatrix.Pitch_FFT_SD[index] = pitchFFTstd

        yawFFT = np.fft.fft(timeFrame.selected_navdata[:, navdataTranslate['Yaw']])
        yawFFtmean = np.mean(yawFFT)
        dataMatrix.Yaw_FFT_Mean_Real[index] = yawFFtmean.real
        dataMatrix.Yaw_FFT_Mean_Imag[index] = yawFFtmean.imag
        yawFFTstd = np.std(yawFFT)
        dataMatrix.Yaw_FFT_SD[index] = yawFFTstd

        index += 1

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




'''
random.seed(33) #does not seem to work reliably.
random.shuffle(labelled) #train/test split

predicted_variable=1 #which of the labels should be predicted?

convert_label ={"-0.25":0, "0.0":1,"0.25":2}
labels = np.array(list(map(lambda x:convert_label[ x.most_common_commands[predicted_variable]], labelled)))

data  =np.array(list(map(lambda x: x.aggregated_navdata, labelled)))
selected_columns=np.array([11,12,13],dtype=np.int32)

data_train=data[:100,  selected_columns]
data_test =data[ 100:, selected_columns]

labels_train=labels[:100]
labels_test =labels[ 100:]


tree=sklearn.tree.DecisionTreeClassifier()
tree=tree.fit(data_train,labels_train)
predicted=tree.predict(data_test)
confussion_matrix=np.zeros([3,3])

for i in range(0,predicted.shape[0]):
    confussion_matrix[labels_test[i]][predicted[i]]+=1

print(confussion_matrix)
'''

'''
disp = plot_confusion_matrix(tree, data_test, labels_test,
                             display_labels=[*frozenCmds.values()],
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title("super cm")
print('super cm')
print(disp.confusion_matrix)
plt.show()
plt.close()
'''


#Confussion matrices:
#Row: true label, Column: predicted label
#(one of the achieved results. seed does not seem to work properly,
#so another run will likely give different results)

#predicted_variable=0
#[[ 1.  0.  0.]
# [ 0. 57.  0.]
# [ 0.  0.  0.]]
#
#predicted_variable=1
#[[15.  1.  3.]
# [ 1. 15.  1.]
# [ 3.  2. 17.]]
#
#predicted_variable=2
#[[ 2.  1.  0.]
# [ 2. 53.  0.]
# [ 0.  0.  0.]]
#
#predicted_variable=3
#[[ 0.  0.  0.]
# [ 0. 45.  2.]
# [ 0.  0. 11.]]