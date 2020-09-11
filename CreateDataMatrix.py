import pandas as pd
import numpy as np
from frozendict import frozendict
from ReadResampleMerge import merged
from ReadResampleMerge import mergedCM
from ReadResampleMerge import mergedSecondSet
from ReadResampleMerge import mergedSecondCM
from ReadResampleMerge import mergedLeftRight

# this modul turn resampled and merged into a matrices go base a machine learning on

def CreateEmptyDataFrame(intervalLen, intIndexedDF, dataColumnNames):
    '''
    creates an empty dataFrame with columns given by dataColumnNames
    that has one row for each interval of intervalLen of records from merged dataframe (intIndexedDF)
    indexed by starting point of time intervals of intervalLen
    :param intervalLen: how many records from inputDF creates one interval (therefore one record of output dataframe)
    :param intIndexedDF: output of ReadResampleMerge with time index reset so that its indexed by ints
    :param dataColumnNames: array containing names of outputDF columns
    :return: empty time indexed dataFrame with appropriate num of columns and rows
    '''
    # prepare indices for a new dataframe
    ind = list()
    indexIterator = 0  # int, counts rows
    while indexIterator < intIndexedDF.index.size:
        # time from the row corresponding to index iterator - STARTING point of the interval
        ind.append(intIndexedDF.time.iloc[indexIterator])
        # move by intervalLen number of rows ahead
        indexIterator += intervalLen
    #print(ind)
    #print(indexIterator)

    # create an empty dataframe
    dataForDT = pd.DataFrame(columns=dataColumnNames, index=ind)
    # interpret int as a time (it is a time)
    dataForDT.index = pd.to_datetime(dataForDT.index, unit='ms')
    dataForDT.index.name = 'time'
    return dataForDT


def CreateEmptyDataFrameWithShift(intervalLen, intIndexedDF, dataColumnNames, representantShift):
    """
    creates an empty dataFrame with columns given by dataColumnNames
    that has one row for each interval of intervalLen of records from merged dataframe (intIndexedDF)
    indexed by a time of nth point in interval where n=representatntShift
    :param intervalLen: how many records from inputDF creates one interval (therefore one record of output dataframe)
    :param intIndexedDF: output of ReadResampleMerge with time index reset so that its indexed by ints
    :param dataColumnNames: array containing names of outputDF columns
    :param representantShift: which sample should represent an interval - by time and cmds
    :return:
    """
    if representantShift >= intervalLen:
        # trying to represent an interval by a sample that does not belong to it
        raise AttributeError
    # prepare indices for a new dataFrame
    indexOfNexDF = list()
    startIdicesOfIntervals = list()
    indexIterator = 0  # int, counts rows
    while (indexIterator + representantShift) < intIndexedDF.index.size:
        # time from the row corresponding to index iterator - representantShift'th point of interval
        indexOfNexDF.append(intIndexedDF.time.iloc[indexIterator+representantShift])
        startIdicesOfIntervals.append(intIndexedDF.time.iloc[indexIterator])
        # move by intervalLen number of rows ahead
        indexIterator += intervalLen
    #print(indexOfNexDF)
    #print(indexIterator)

    # create an empty dataframe
    dataForDT = pd.DataFrame(columns=dataColumnNames, index=indexOfNexDF)
    # interpret int as a time (it is a time)
    dataForDT.index = pd.to_datetime(dataForDT.index, unit='ms')
    dataForDT.index.name = 'time'
    return dataForDT, startIdicesOfIntervals


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
    works - use this function to obtain discrete vals of cmds
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
    fill one row of dataFrame for MLmodel with data
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
    fill one row of dataFrame for MLmodel with data
    for roll, pitch and jaw count mean, std and mean and std of fft
    each row represents one interval of intervalLen records from inputDF
    split complex values to real and imaginary part
    :param index: index of the row of outputDF that is to be filled with data by this call
    :param indexIterator: index of the row of inputDF - the first row of the interval
    :param intervalLen: how many records does one interval contain
    :param inputDF: output of ReadresampleMerge - data to create dataMatrix for ML algs. from
    :param outputDF: data matrix to base machine learning on
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


def CreateData(index, indexIterator, intervalLen, inputDF, outputDF, namesAndFunc):
    """
    fill one row of dataFrame for MLmodel with data
    functions to count values of outputDF based on interval of values from inputDF are passed in nameAndFunc array od dictionaries
    each dictionary has to have keys 'Name', 'Func', 'Column'
    each row represents one interval of intervalLen records from inputDF
    :param index:
    :param indexIterator:
    :param intervalLen:
    :param inputDF:
    :param outputDF:
    :param namesAndFunc: namesAndFunc = [{'Name': 'Name of column in outputDF to be filled', 'Func': func to count a value, 'Column': column of inputDF to provide args for a function },...]
    """
    for record in namesAndFunc:
        outputColumnName = record['Name']
        function = record['Func']
        inputColumnName = record['Column']
        outputDF[outputColumnName][index] = function(inputDF[inputColumnName][indexIterator:indexIterator + intervalLen])


def CreateDataFrameForDTMatrix(inputDFmerged, ColumnNames, functionToCreateContend=CreateDataWithRealAndImagPart,
                               functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict, intervalLen=40):
    '''
    splits rows to intervalLen long intervals, counts mean, std and mean and std of fft within these intervals
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


def CreateDataFrameForDTMatrixShift(inputDFmerged, intervalLen=40, representantSampleShift=0, funcArrayToCreateContent=None, ColumnNames=None):
    """
    splits rows to intervalLen long intervals
    when funcArrayToCreateContent is None counts mean, std and mean and std of fft within these intervals
    otherwise counts values using function passed in funcArrayToCreateContent
    creates a new dataFrame with one row for each interval
    time index of row is the time of the nth record in the interval where n=representantSampleShift
    :param inputDFmerged: aoutput of ReadResampleMerge
    :param intervalLen:
    :param representantSampleShift:
    :param funcArrayToCreateContent:
    :param ColumnNames:
    :return: data matrix to base machine learning on
    """
    intIndexed = inputDFmerged.reset_index()
    if ColumnNames is None:
        ColumnNames = dataColumnNamesRealImag
    newDF, startOfIntervalIndices = CreateEmptyDataFrameWithShift(intervalLen, intIndexed, ColumnNames, representantSampleShift)

    # fill dataFrame with data
    if funcArrayToCreateContent is not None:
        # use functions from funcArrayToCreateContent to create content of DF
        indexIterator = 0
        for index in newDF.index:
            MakeCMDsDiscreteWithFrozenDict(index, inputDFmerged, newDF)
            CreateData(index, indexIterator, intervalLen, inputDF=inputDFmerged, outputDF=newDF, namesAndFunc=funcArrayToCreateContent)
            indexIterator += intervalLen
    else:
        # count mean, std and mean and std of fft as a content of DF
        indexIterator = 0
        for index in newDF.index:
            MakeCMDsDiscreteWithFrozenDict(index, inputDFmerged, newDF)
            CreateDataWithRealAndImagPart(index, indexIterator, intervalLen, inputDF=inputDFmerged, outputDF=newDF)
            indexIterator += intervalLen

    return newDF


# Create DataForDTRealImag
dataColumnNamesRealImag = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean_Real', 'Roll_FFT_Mean_Imag',
                           'Roll_FFT_SD', 'Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean_Real', 'Pitch_FFT_Mean_Imag', 'Pitch_FFT_SD',
                           'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean_Real', 'Yaw_FFT_Mean_Imag', 'Yaw_FFT_SD']

# Create Data fot DT with separated real and imag part and cmds as frozen set item
frozenCmds = frozendict({1: '-', 2: '0', 3: '+'})
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

# data matrix from newer own data
dataDTSecondSet = CreateDataFrameForDTMatrix(inputDFmerged=mergedSecondSet, ColumnNames=dataColumnNamesRealImag,
                                       functionToCreateContend=CreateDataWithRealAndImagPart,
                                       functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                       intervalLen=40)
dataDTSecondSet.to_csv('OutputStages\\dataDTSecondSet.tsv', sep='\t')

# data matrix for model evaluation from own data
dataDTSecondCM = CreateDataFrameForDTMatrix(inputDFmerged=mergedSecondCM, ColumnNames=dataColumnNamesRealImag,
                                       functionToCreateContend=CreateDataWithRealAndImagPart,
                                       functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                       intervalLen=40)
dataDTSecondCM.to_csv('OutputStages\\dataDTSecondCM.tsv', sep='\t')


# USELESS - too few samples
leftRightData = CreateDataFrameForDTMatrix(inputDFmerged=mergedLeftRight, ColumnNames=dataColumnNamesRealImag,
                                       functionToCreateContend=CreateDataWithRealAndImagPart,
                                       functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict,
                                       intervalLen=20)
leftRightData.to_csv('OutputStages\\leftRightData.tsv', sep='\t')

# to test CreateDataFrameForDTMatrixShift function
shifted = CreateDataFrameForDTMatrixShift(inputDFmerged=merged, intervalLen=40,representantSampleShift=5)
shifted.to_csv('OutputStages\\shifted5Data.tsv', sep='\t')

# example of passing own functions to create dataFrame contend to CreateDataFrameForDTMatrixShift method
namesAndFunc = [{'Name': 'Roll_Mean', 'Func': np.mean, 'Column': 'Roll_x'}, {'Name': 'Roll_SD', 'Func': np.std, 'Column': 'Roll_x'},
                {'Name': 'Pitch_Mean', 'Func': np.mean, 'Column': 'Pitch_y'}, {'Name': 'Pitch_SD', 'Func': np.std, 'Column': 'Pitch_y'},
                {'Name': 'Yaw_Mean', 'Func': np.mean, 'Column': 'Yaw_z'}, {'Name': 'Yaw_SD', 'Func': np.std, 'Column': 'Yaw_z'}]
dataColumnNamesFucParams = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD',  'Pitch_Mean', 'Pitch_SD',
                           'Yaw_Mean', 'Yaw_SD']
funcParamsData = CreateDataFrameForDTMatrixShift(inputDFmerged=merged, intervalLen=40, representantSampleShift=0,
                                                 funcArrayToCreateContent=namesAndFunc, ColumnNames=dataColumnNamesFucParams)
funcParamsData.to_csv('OutputStages\\funcParamData.tsv', sep='\t')


