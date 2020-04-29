import pandas as pd
import numpy as np
from frozendict import frozendict


def CreateEmptyDataFrame(intervalLen, intIndexedDF, dataColumnNames):
    # create matrix to base machine learning on
    # intervalLen = 40
    # prepare indices for a new dataframe
    ind = list()
    indexIterator = 0
    while indexIterator < intIndexedDF.index.size:
        ind.append(intIndexedDF.time.iloc[indexIterator])
        indexIterator += intervalLen
    #print(ind)
    #print(indexIterator)

    # create an empty dataframe
    dataForDT = pd.DataFrame(columns=dataColumnNames, index=ind)
    dataForDT.index = pd.to_datetime(dataForDT.index, unit='ms')
    dataForDT.index.name = 'time'
    return dataForDT

def MakeCMDsDiscrete(index, inputDF, outputDF):
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


def CreateDataFrame(inputDFmerged, ColumnNames, functionToCreateContend, functionToDiscreteCmds, intervalLen=40):
    intIndexed = inputDFmerged.reset_index()
    newDF = CreateEmptyDataFrame(intervalLen, intIndexed, ColumnNames)

    indexIterator = 0
    for index in newDF.index:
        functionToDiscreteCmds(index, inputDFmerged, newDF)
        functionToCreateContend(index, indexIterator, intervalLen, inputDF=inputDFmerged, outputDF=newDF)
        indexIterator += intervalLen

    return newDF


cmdsColumnNames = ['time', 'leftRight', 'frontBack', 'up', 'angular']
cmds = pd.read_csv('InputData\\commands.tsv', delimiter='\t', header=None, names=cmdsColumnNames)
cmds.time = pd.to_datetime(cmds.time, unit='ms')
cmds = cmds.set_index('time')
# print("unique cmds time:", cmds.index.is_unique)

resampledCmds = cmds.resample('50ms').mean()
resampledCmds = resampledCmds.reindex().bfill()
resampledCmds = resampledCmds[cmds.index[0]:cmds.index[-1]]

# resampledCmds.to_csv('resampledCmds.tsv', sep='\t')

navDataColumnNames = ['time', 'State', 'Battery_level', 'Magnetometer_x', 'Magnetometer_y', 'Magnetometer_z', 'Pressure',
                      'Temperature', 'Wind_speed', 'Wind_angle', 'Wind_compensation:pitch', 'Wind_compensation:roll',
                      'Pitch_y', 'Roll_x', 'Yaw_z', 'Altitude', 'Velocity_x',
                      'Velocity_y', 'Velocity_z', 'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
                      'Motor_1_power', 'Motor_2_power', 'Motor_3_power', 'Motor_4_power', 'time_board']

navData = pd.read_csv('InputData\\navdata.tsv', delimiter='\t', header=None, usecols=range(1, 28), names=navDataColumnNames)
# print("unique navdata time:", navData.time.is_unique)
#get rid of duplicated times in navData
navData.drop_duplicates(['time'], keep='first', inplace=True)
# print("unique navdata time after .drop_duplicated call:", navData.time.is_unique)
navData.time = pd.to_datetime(navData.time, unit='ms')
navData = navData.set_index('time')

resampledNav = navData.resample('50ms').mean()
resampledNav = resampledNav.reindex().bfill()
resampledNav = resampledNav[cmds.index[0]:cmds.index[-1]]

# resampledNav.to_csv('resampledNav.tsv', sep='\t')

# how inner - intersection, keeps only times that do have corresponding counterpart in second file
merged = pd.merge(resampledCmds, resampledNav, right_index=True, left_index=True, how='inner') #merge when time is index
#inputDF = pd.merge(cmds, navData, right_on="time", left_on="time", how='inner', indicator=True)

# merged.to_csv("mergedResampled.tsv", sep='\t')
# print("unique time right after merge:", merged.index.is_unique)

#TODO: after this outputDF is indexed by senceless dates begginning with start of unix date - does it make any sence?
#TODO: can I somehow format string representation of values in particular columns while printing them to file by to_scv
#TODO: to print 'time' in some meaningfull format

# Create DataForDTComplex
dataColumnNamesComplex = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean', 'Roll_FFT_SD',
                   'Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean', 'Pitch_FFT_SD', 'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean',
                   'Yaw_FFT_SD']
dataForDTComplex = CreateDataFrame(inputDFmerged=merged, ColumnNames=dataColumnNamesComplex,
                                   functionToCreateContend=CreateDataWithComplexValues, functionToDiscreteCmds=MakeCMDsDiscrete, intervalLen=40)
dataForDTComplex.to_csv('OutputStages\\dataForDTComplex.tsv', sep='\t')

# Create DataForDTRealImag
dataColumnNamesRealImag = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean_Real', 'Roll_FFT_Mean_Imag',
                           'Roll_FFT_SD','Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean_Real', 'Pitch_FFT_Mean_Imag', 'Pitch_FFT_SD',
                           'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean_Real', 'Yaw_FFT_Mean_Imag','Yaw_FFT_SD']
dataForDTRealImag = CreateDataFrame(inputDFmerged=merged, ColumnNames=dataColumnNamesRealImag, functionToCreateContend=CreateDataWithRealAndImagPart,
                                    functionToDiscreteCmds=MakeCMDsDiscrete, intervalLen=40)
dataForDTRealImag.to_csv('OutputStages\\dataForDTRealImag.tsv', sep='\t')

# Create Data fot DT with separated real and imag part and cms as enum
dataForDTRealImagEnum = CreateDataFrame(inputDFmerged=merged, ColumnNames=dataColumnNamesRealImag, functionToCreateContend=CreateDataWithRealAndImagPart,
                                    functionToDiscreteCmds=MakeCMDsDiscreteWithEnum, intervalLen=40)

print("DESCRIBING")
print(dataForDTRealImagEnum.describe())
dataForDTRealImagEnum.to_csv('OutputStages\\dataForDTRealImagEnum.tsv', sep='\t')

# Create Data fot DT with separated real and imag part and cms as frozen set item
frozenCmds = frozendict({1: '-', 2: '0', 3: '+'})
dataForDTRealImagFrozenDict = CreateDataFrame(inputDFmerged=merged, ColumnNames=dataColumnNamesRealImag, functionToCreateContend=CreateDataWithRealAndImagPart,
                                    functionToDiscreteCmds=MakeCMDsDiscreteWithFrozenDict, intervalLen=40)
dataForDTRealImagFrozenDict.to_csv('OutputStages\\dataForDTRealImagFrozenDict.tsv', sep='\t')




