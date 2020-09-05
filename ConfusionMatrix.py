import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

import BuildDT as bdt
from CreateDataMatrixForDT import dataForCM
from CreateDataMatrixForDT import frozenCmds
from CreateDataMatrixForDT import dataDTSecondCM


# TODO: standartisation of cm data? if so, how?
def CreateConfusionMatrix(labelColumnName, validationDataMatrix, decisionTree, cmTitle, color):
    '''
    for given decision tree creates a confusion matrix showing how well DT performs while predicting
    labels from validationDataMatrix Dt was not trained on
    :param labelColumnName: name of a label column that is predicted bt DT
    :param validationDataMatrix: data matrix, that DT was not trained on
    :param decisionTree: DT whose quality is to be evaluated
    :param cmTitle: name fot an output confusion matrix
    :return:
    '''
    realLabels = bdt.GetLabel(labelColumnName, validationDataMatrix)
    features = bdt.GetFeatures(validationDataMatrix)
    predictedLabels = decisionTree.predict(features)
    # if labelColumnName == 'leftRight':
    #    features = bdt.scaler.transform(features)

    disp = plot_confusion_matrix(decisionTree, features, realLabels,
                                 display_labels=[*frozenCmds.values()],
                                 cmap=color,
                                 normalize=None)
    disp.ax_.set_title(cmTitle)
    print(cmTitle)
    print(disp.confusion_matrix)
    plt.show()



CreateConfusionMatrix('leftRight', dataForCM, bdt.DTleftRight, "leftRightCM", plt.cm.Blues)
CreateConfusionMatrix('frontBack', dataForCM, bdt.DTfrontBack, "frontBackCM", plt.cm.Blues)
CreateConfusionMatrix('angular', dataForCM, bdt.DTangular, "angularCM", plt.cm.Blues)

CreateConfusionMatrix('leftRight', dataDTSecondCM, bdt.DTleftRightSecond, "leftRightCMSecond", plt.cm.Reds)
CreateConfusionMatrix('frontBack', dataDTSecondCM, bdt.DTfrontBackSecond, "frontBackCMSecond", plt.cm.Reds)
CreateConfusionMatrix('angular', dataDTSecondCM, bdt.DTangularSecond, "angularCMSecond", plt.cm.Reds)

'''
realLabels = bdt.GetLabel("leftRight", dataForCM)
features = bdt.GetFeatures(dataForCM)
predictedLabels = bdt.DTleftRight.predict(features)

# print(bdt.x_testLR)
# y_pred = bdt.DTleftRight.predict(bdt.x_testLR)
# print("y_pred:")
# print(y_pred)
# print("y_testLR")
# print(bdt.y_testLR)
# cm = confusion_matrix(realLabels, predictedLabels)
# print(cm)

#plot_confusion_matrix(cm, normalize=False, target_names=[*frozenCmds.values()], title="Confusion Matrix")

disp = plot_confusion_matrix(bdt.DTleftRight, features, realLabels,
                                 display_labels=[*frozenCmds.values()],
                                 cmap=plt.cm.Blues,
                                 normalize=None)
disp.ax_.set_title("My awesome CM")
print("My awesome CM")
print(disp.confusion_matrix)

plt.show()
'''