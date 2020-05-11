import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

import BuildDT as bdt
from CreateDataMatrixForDT import dataForCM
from CreateDataMatrixForDT import frozenCmds


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
