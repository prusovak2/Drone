import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

import BuildDT as bdt
from CreateDataMatrix import dataForCM
from CreateDataMatrix import frozenCmds
from CreateDataMatrix import dataDTSecondCM


def CreateConfusionMatrix(labelColumnName, validationDataMatrix, MLmodel, cmTitle, color, scaler=None, showCM=True):
    '''
    for given MLmodel creates a confusion matrix showing how well the model performs while predicting
    labels from validationDataMatrix model was not trained on
    :param labelColumnName: name of a label column that is predicted by model
    :param validationDataMatrix: data matrix, that model was not trained on
    :param MLmodel: model whose quality is to be evaluated
    :param cmTitle: name of an output confusion matrix
    :param color: color of created CM, plt.cm.something
    :param scaler: standardSacler that was used to standardize training data for the model, is used to standardize validationData as well
    :param showCM: should CM be shown?
    :return: features of validation data, standardized when scaler is not None
    '''
    realLabels = bdt.GetLabel(labelColumnName, validationDataMatrix)
    features = bdt.GetFeatures(validationDataMatrix)
    if scaler is not None:
        # standardize validation data by same scaler that was used for a training data
        features = scaler.transform(features)

    disp = plot_confusion_matrix(MLmodel, features, realLabels,
                                 display_labels=[*frozenCmds.values()],
                                 cmap=color,
                                 normalize=None)
    disp.ax_.set_title(cmTitle)
    print(cmTitle)
    print(disp.confusion_matrix)
    if showCM is True:
        plt.show()
    return features


if __name__ == "__main__":
    CreateConfusionMatrix('leftRight', dataForCM, bdt.DTleftRight, "leftRightCM", plt.cm.Blues)
    CreateConfusionMatrix('frontBack', dataForCM, bdt.DTfrontBack, "frontBackCM", plt.cm.Blues)
    CreateConfusionMatrix('angular', dataForCM, bdt.DTangular, "angularCM", plt.cm.Blues)

    CreateConfusionMatrix('leftRight', dataDTSecondCM, bdt.DTleftRightSecond, "leftRightCMSecond", plt.cm.Reds)
    CreateConfusionMatrix('frontBack', dataDTSecondCM, bdt.DTfrontBackSecond, "frontBackCMSecond", plt.cm.Reds)
    CreateConfusionMatrix('angular', dataDTSecondCM, bdt.DTangularSecond, "angularCMSecond", plt.cm.Reds)

