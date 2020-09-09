import pandas as pd
import numpy as np
from frozendict import frozendict
from ReadResampleMerge import merged
from ReadResampleMerge import mergedCM
from ReadResampleMerge import mergedSecondSet
from ReadResampleMerge import mergedSecondCM
from ReadResampleMerge import mergedLeftRight

def CreateEmptyDataFrame(intervalLen, intIndexedDF, dataColumnNames):
    '''
    create matrix to base a machine learning on
    indexed starting point of time intervals of intervalLen
    :param intervalLen:
    :param intIndexedDF:
    :param dataColumnNames:
    :return: empty indexed dataFrame
    '''
    # intervalLen = 40
    # prepare indices for a new dataframe
    ind = list()
    indexIterator = 0 # int, counts rows
    while indexIterator < intIndexedDF.index.size:
        # time from the row corresponding to index iterator - STARTING point of the interval
        ind.append(intIndexedDF.time.iloc[indexIterator])
        # move by intervalLen number of rows ahead
        indexIterator += intervalLen
    #print(ind)
    #print(indexIterator)

    # create an empty dataframe
    dataForDT = pd.DataFrame(columns=dataColumnNames, index=ind)
    # interpret ind as a time (it is a time)
    dataForDT.index = pd.to_datetime(dataForDT.index, unit='ms')
    dataForDT.index.name = 'time'
    return dataForDT


def MakeCMDsDiscrete(index, inputDF, outputDF):
    '''
    gain discrete value of CMDs on given index
    doesn't work properly
    :param index:
    :param inputDF:
    :param outputDF:
    :return:
    '''
    #cmds
    #leftRight
    if inputDF.leftRight[index] <= -0.05:
        outputDF.leftRight[index] = 1
    elif inputDF.leftRight[index] >= 0.05:
        outputDF.leftRight[index] = 3
    else:
        outputDF.leftRight[index] = 2

    #frontBack
    if inputDF.frontBack[index] <= -0.05:
        outputDF.frontBack[index] = 1
    elif inputDF.frontBack[index] >= 0.05:
        outputDF.frontBack[index] = 3
    else:
        outputDF.frontBack[index] = 2

    #angular
    if inputDF.angular[index] <= -0.05:
        outputDF.angular[index] = 1
    elif inputDF.angular[index] >= 0.05:
        outputDF.angular[index] = 3
    else:
        outputDF.angular[index] = 2


from enum import Enum


class Cmd(Enum):
    Pos = 1
    RoundZero = 2
    Neg = 3


def MakeCMDsDiscreteWithEnum(index, inputDF, outputDF):
    '''
    gain discrete value of CMDs on given index
    discrete values represented as members of enum
    doesn't work properly
    :param index:
    :param inputDF:
    :param outputDF:
    :return:
    '''
    # cmds
    # leftRight
    if inputDF.leftRight[index] <= -0.05:
        outputDF.leftRight[index] = Cmd.Neg
    elif inputDF.leftRight[index] >= 0.05:
        outputDF.leftRight[index] = Cmd.Pos
    else:
        outputDF.leftRight[index] = Cmd.RoundZero

    # frontBack
    if inputDF.frontBack[index] <= -0.05:
        outputDF.frontBack[index] = Cmd.Neg
    elif inputDF.frontBack[index] >= 0.05:
        outputDF.frontBack[index] = Cmd.Pos
    else:
        outputDF.frontBack[index] = Cmd.RoundZero

    # angular
    if inputDF.angular[index] <= -0.05:
        outputDF.angular[index] = Cmd.Neg
    elif inputDF.angular[index] >= 0.05:
        outputDF.angular[index] = Cmd.Pos
    else:
        outputDF.angular[index] = Cmd.RoundZero


def MakeCMDsDiscreteWithFrozenDict(index, inputDF, outputDF):
    '''
    USE THIS
    gain discrete value of CMDs on given index
    discrete values represented as values from frozen dictionary
    works
    :param index:
    :param inputDF:
    :param outputDF:
    :return:
    '''
    # cmds
    # leftRight
    if inputDF.leftRight[index] <= -0.05:
        outputDF.leftRight[index] = frozenCmds[1]
    elif inputDF.leftRight[index] >= 0.05:
        outputDF.leftRight[index] = frozenCmds[3]
    else:
        outputDF.leftRight[index] = frozenCmds[2]

    # frontBack
    if inputDF.frontBack[index] <= -0.05:
        outputDF.frontBack[index] = frozenCmds[1]
    elif inputDF.frontBack[index] >= 0.05:
        outputDF.frontBack[index] = frozenCmds[3]
    else:
        outputDF.frontBack[index] = frozenCmds[2]

    # angular
    if inputDF.angular[index] <= -0.05:
        outputDF.angular[index] = frozenCmds[1]
    elif inputDF.angular[index] >= 0.05:
        outputDF.angular[index] = frozenCmds[3]
    else:
        outputDF.angular[index] = frozenCmds[2]


def CreateDataWithComplexValues(index, indexIterator, intervalLen, inputDF, outputDF):
    '''
    fill one row of dataFrame for decisionTree with data
    for roll, pitch and jaw count mean, std and mean and std of fft
    each row represents one interval of intervalLen records from inputDF
    keep complex values in outputDF
    :param index:
    :param indexIterator:
    :param intervalLen:
    :param inputDF:
    :param outputDF:
    :return:
    '''
    # Roll
    outputDF.Roll_Mean[index] = np.mean(inputDF.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Roll_SD[index] = np.std(inputDF.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(inputDF.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Roll_FFT_Mean[index] = np.mean(fft)
    outputDF.Roll_FFT_SD[index] = np.std(fft)
    # Pitch
    outputDF.Pitch_Mean[index] = np.mean(inputDF.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Pitch_SD[index] = np.std(inputDF.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(inputDF.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Pitch_FFT_Mean[index] = np.mean(fft)
    outputDF.Pitch_FFT_SD[index] = np.std(fft)
    # Yaw
    outputDF.Yaw_Mean[index] = np.mean(inputDF.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Yaw_SD[index] = np.std(inputDF.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(inputDF.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Yaw_FFT_Mean[index] = np.mean(fft)
    outputDF.Yaw_FFT_SD[index] = np.std(fft)


def CreateDataWithRealAndImagPart(index, indexIterator, intervalLen, inputDF, outputDF):
    '''
    fill one row of dataFrame for decisionTree with data
    for roll, pitch and jaw count mean, std and mean and std of fft
    each row represents one interval of intervalLen records from inputDF
    split complex values to real and imaginary part
    :param index:
    :param indexIterator:
    :param intervalLen:
    :param inputDF:
    :param outputDF:
    :return:
    '''
    # Roll
    outputDF.Roll_Mean[index] = np.mean(inputDF.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Roll_SD[index] = np.std(inputDF.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(inputDF.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    RollFFTMean = np.mean(fft)
    outputDF.Roll_FFT_Mean_Real[index] = RollFFTMean.real
    outputDF.Roll_FFT_Mean_Imag[index] = RollFFTMean.imag
    outputDF.Roll_FFT_SD[index] = np.std(fft)
    # Pitch
    outputDF.Pitch_Mean[index] = np.mean(inputDF.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Pitch_SD[index] = np.std(inputDF.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(inputDF.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    PitchFFTMean = np.mean(fft)
    outputDF.Pitch_FFT_Mean_Real[index] = PitchFFTMean.real
    outputDF.Pitch_FFT_Mean_Imag[index] = PitchFFTMean.imag
    outputDF.Pitch_FFT_SD[index] = np.std(fft)
    # Yaw
    outputDF.Yaw_Mean[index] = np.mean(inputDF.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    outputDF.Yaw_SD[index] = np.std(inputDF.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(inputDF.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    YawFFTMean = np.mean(fft)
    outputDF.Yaw_FFT_Mean_Real[index] = YawFFTMean.real
    outputDF.Yaw_FFT_Mean_Imag[index] = YawFFTMean.imag
    outputDF.Yaw_FFT_SD[index] = np.std(fft)


def CreateDataFrameForDTMatrix(inputDFmerged, ColumnNames, functionToCreateContend, functionToDiscreteCmds, intervalLen=40):
    '''
    splits rows to intervalLen long intervals, counts mean, std and mean and std of ffr within these intervals
    creates a new dataFrame with one row for each interval, time index of row is the time of the first record in the interval
    :param inputDFmerged:
    :param ColumnNames:
    :param functionToCreateContend:
    :param functionToDiscreteCmds:
    :param intervalLen:
    :return: dataFrame to base decision tree on
    '''
    intIndexed = inputDFmerged.reset_index()
    newDF = CreateEmptyDataFrame(intervalLen, intIndexed, ColumnNames)

    # fill dataFrame with data
    indexIterator = 0
    for index in newDF.index:
        functionToDiscreteCmds(index, inputDFmerged, newDF)
        functionToCreateContend(index, indexIterator, intervalLen, inputDF=inputDFmerged, outputDF=newDF)
        indexIterator += intervalLen

    return newDF




'''
# Create DataForDTComplex
dataColumnNamesComplex = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean', 'Roll_FFT_SD',
                   'Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean', 'Pitch_FFT_SD', 'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean',
                   'Yaw_FFT_SD']
dataForDTComplex = CreateDataFrameForDTMatrix(inputDFmerged=merged, ColumnNames=dataColumnNamesComplex,
                                              functionToCreateContend=CreateDataWithComplexValues, functionToDiscreteCmds=MakeCMDsDiscrete, intervalLen=40)
dataForDTComplex.to_csv('OutputStages\\dataForDTComplex.tsv', sep='\t')
'''
# Create DataForDTRealImag
dataColumnNamesRealImag = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean_Real', 'Roll_FFT_Mean_Imag',
                           'Roll_FFT_SD','Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean_Real', 'Pitch_FFT_Mean_Imag', 'Pitch_FFT_SD',
                           'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean_Real', 'Yaw_FFT_Mean_Imag','Yaw_FFT_SD']
'''
dataForDTRealImag = CreateDataFrameForDTMatrix(inputDFmerged=merged, ColumnNames=dataColumnNamesRealImag, functionToCreateContend=CreateDataWithRealAndImagPart,
                                               functionToDiscreteCmds=MakeCMDsDiscrete, intervalLen=40)
dataForDTRealImag.to_csv('OutputStages\\dataForDTRealImag.tsv', sep='\t')

# Create Data fot DT with separated real and imag part and cms as enum
dataForDTRealImagEnum = CreateDataFrameForDTMatrix(inputDFmerged=merged, ColumnNames=dataColumnNamesRealImag, functionToCreateContend=CreateDataWithRealAndImagPart,
                                                   functionToDiscreteCmds=MakeCMDsDiscreteWithEnum, intervalLen=40)

# print("DESCRIBING")
# print(dataForDTRealImagEnum.describe())
dataForDTRealImagEnum.to_csv('OutputStages\\dataForDTRealImagEnum.tsv', sep='\t')
'''

# Create Data fot DT with separated real and imag part and cmds as frozen set item
frozenCmds = frozendict({1: '-', 2: '0', 3: '+'})
# Original data
# create dataFrame to base a decision tree on
dataForDTRealImagFrozenDict = CreateDataFrameForDTMatrix(inputDFmerged=merged, ColumnNames=dataColumnNamesRealImag,
                                                         functionToCreateContend=CreateDataWithRealAndImagPart,
                                                         functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                                         intervalLen=40)
dataForDTRealImagFrozenDict.to_csv('OutputStages\\dataForDTRealImagFrozenDict.tsv', sep='\t')
# new data for Confusion Matrix - evaluation of DT
dataForCM = CreateDataFrameForDTMatrix(inputDFmerged=mergedCM, ColumnNames=dataColumnNamesRealImag,
                                       functionToCreateContend=CreateDataWithRealAndImagPart,
                                       functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                       intervalLen=40)
dataForCM.to_csv('OutputStages\\dataForCM.tsv', sep='\t')

dataDTSecondSet = CreateDataFrameForDTMatrix(inputDFmerged=mergedSecondSet, ColumnNames=dataColumnNamesRealImag,
                                       functionToCreateContend=CreateDataWithRealAndImagPart,
                                       functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                       intervalLen=40)
dataDTSecondSet.to_csv('OutputStages\\dataDTSecondSet.tsv', sep='\t')

dataDTSecondCM = CreateDataFrameForDTMatrix(inputDFmerged=mergedSecondCM, ColumnNames=dataColumnNamesRealImag,
                                       functionToCreateContend=CreateDataWithRealAndImagPart,
                                       functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                       intervalLen=40)
dataDTSecondSet.to_csv('OutputStages\\dataDTSecondCM.tsv', sep='\t')

leftRightData = CreateDataFrameForDTMatrix(inputDFmerged=mergedLeftRight, ColumnNames=dataColumnNamesRealImag,
                                       functionToCreateContend=CreateDataWithRealAndImagPart,
                                       functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                       intervalLen=40)
dataDTSecondSet.to_csv('OutputStages\\leftRightData.tsv', sep='\t')


