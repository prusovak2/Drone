import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import BuildDT as bdt
from CreateDataMatrixForDT import dataForCM
realLabels = bdt.GetLabel("leftRight", dataForCM)
features = bdt.GetFeatures(dataForCM)
predictedLabels = bdt.DTleftRight.predict(features)

# print(bdt.x_testLR)
# y_pred = bdt.DTleftRight.predict(bdt.x_testLR)
# print("y_pred:")
# print(y_pred)
# print("y_testLR")
# print(bdt.y_testLR)
cm = confusion_matrix(realLabels, predictedLabels)
print(cm)

