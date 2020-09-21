import pandas as pd
import numpy as np

# this modul enables to read CMD and NAVDATA files, resample them to given frequency and merge them together
# it provides a preparation for data preprocessing carried out by CreateDataMatrixFotDT modul

def ReadCMDScvsIntoDFandResample(filename, frequency):
    '''
    reads CMD file and resamples it to given frequency
    samples are resampled so that their time indices create an arithmetic sequence with common difference of frequency parameter
    :param filename: file to read commands from
    :param frequency: ms
    :return: dataFrame containing resampled CMDs
    '''
    cmdsColumnNames = ['time', 'leftRight', 'frontBack', 'up', 'angular']
    cmds = pd.read_csv(filename, delimiter='\t', header=None, names=cmdsColumnNames)
    cmds.time = pd.to_datetime(cmds.time, unit='ms')
    cmds = cmds.set_index('time')
    # print("unique cmds time:", cmds.index.is_unique)

    # resample
    resampledCmds = cmds.resample(frequency).mean()
    resampledCmds = resampledCmds.reindex().bfill()
    resampledCmds = resampledCmds[cmds.index[0]:cmds.index[-1]]
    return resampledCmds

def ReadNAVDATAcsvIntoDFandResample(filename, frequency):
    '''
    reads NAVDATA file and resamples it to given frequency
    samples are resampled so that their time indices create an arithmetic sequence with common difference of frequency parameter
    :param filename: file to read NAVDATA from
    :param frequency:
    :return: dataFrame containing resampled NAVDATA
    '''
    navDataColumnNames = ['time', 'State', 'Battery_level', 'Magnetometer_x', 'Magnetometer_y', 'Magnetometer_z',
                          'Pressure',
                          'Temperature', 'Wind_speed', 'Wind_angle', 'Wind_compensation:pitch',
                          'Wind_compensation:roll',
                          'Pitch_y', 'Roll_x', 'Yaw_z', 'Altitude', 'Velocity_x',
                          'Velocity_y', 'Velocity_z', 'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
                          'Motor_1_power', 'Motor_2_power', 'Motor_3_power', 'Motor_4_power', 'time_board']

    navData = pd.read_csv(filename, delimiter='\t', header=None, usecols=range(1, 28),
                          names=navDataColumnNames)
    # some debug print for adjusting function to new data files
    # suffix = filename[12:]
    # navData.to_csv("OutputStages\\tmpNav"+suffix, sep='\t')
    # print("unique navdata time:", navData.time.is_unique)

    # get rid of duplicated times in navData
    navData.drop_duplicates(['time'], keep='first', inplace=True)
    #print("unique navdata time after .drop_duplicated call:", navData.time.is_unique)
    navData.time = pd.to_datetime(navData.time, unit='ms')
    navData = navData.set_index('time')

    # resample
    resampledNav = navData.resample(frequency).mean()
    resampledNav = resampledNav.reindex().bfill()
    resampledNav = resampledNav[navData.index[0]:navData.index[-1]]
    return resampledNav


def ReadResampleMerge(cmdsFilename, navdataFilename, frequency='50ms', cmdOutputTsvFilename=None, navdataOutputTsvFilename=None,
                      mergedOutputTsvFilename=None):
    '''
    reads cmd and navdata files, resamples them to given frequency and results merges by ctime
    when output file names are given, prints corresponding outputs to .tsv files
    :param cmdsFilename: file to read CMDs from
    :param navdataFilename: file to read NAVDATA from
    :param frequency: ms
    :param cmdOutputTsvFilename: file to output resampled CMDs to
    :param navdataOutputTsvFilename: file to output resampled NAVDARA to
    :param mergedOutputTsvFilename: file to output resampled and merged data
    :return:
    '''
    # READ AND RESAMPLE INPUT DATA
    resampledCmds = ReadCMDScvsIntoDFandResample(cmdsFilename, frequency)
    # print to .tsv file only when given a filename
    if cmdOutputTsvFilename is not None:
        resampledCmds.to_csv(cmdOutputTsvFilename, sep='\t')
    resampledNav = ReadNAVDATAcsvIntoDFandResample(navdataFilename, frequency)
    if navdataOutputTsvFilename is not None:
        resampledNav.to_csv(navdataOutputTsvFilename, sep='\t')

    # preparedCMDS = PrepareCommands(resampledNav,resampledCmds)
    # MERGE INPUT DATA
    # how inner - intersection, keeps only times that do have corresponding counterpart in second file
    merged = pd.merge(resampledCmds, resampledNav, right_index=True, left_index=True,
                      how='inner')  # merge when time is index
    # inputDF = pd.merge(cmds, navData, right_on="time", left_on="time", how='inner', indicator=True)
    if mergedOutputTsvFilename is not None:
        merged.to_csv(mergedOutputTsvFilename, sep='\t')
    # print("unique time right after merge:", merged.index.is_unique)
    return merged

def addTabs(filename, outputFile):
    '''
    adds a column of tabs at the beginning of a file
    :param filename:
    :param outputFile:
    :return:
    '''
    with open(filename, mode='r') as input:
        with open(outputFile, mode='w') as output:
            for line in input:
                output.write('\t'+line)



def PrepareCommands(resampledNav, resampledCMDS):
    columnNames = ['time', 'leftRight', 'frontBack', 'up', 'angular']
    ind = resampledNav.index
    newCommands = pd.DataFrame(columns=columnNames, index=ind)
    currentTime = resampledCMDS.index[0]
    lastSmallerTime = currentTime
    i = 0
    for navTime in ind:
        while currentTime <= navTime:
            lastSmallerTime = currentTime
            i+=1
            if i < resampledCMDS.index.size:
                currentTime = resampledCMDS.index[i]
            else:
                break

        newCommands.leftRight[navTime] = resampledCMDS.leftRight[lastSmallerTime]
        newCommands.frontBack[navTime] = resampledCMDS.frontBack[lastSmallerTime]
        newCommands.up[navTime] = resampledCMDS.up[lastSmallerTime]
        newCommands.angular[navTime] = resampledCMDS.angular[lastSmallerTime]
    return newCommands

'''
cmds =ReadCMDScvsIntoDFandResample('InputData\\commands.tsv','50ms')
nav = ReadNAVDATAcsvIntoDFandResample('InputData\\navdata.tsv', '50ms')
prepCMDS = PrepareCommands(nav, cmds)
prepCMDS.to_csv('OutputStages\\preparedCMDS.tsv',  sep='\t')
'''


# dataForDTRealImagFrozenDict
merged = ReadResampleMerge('InputData\\commands.tsv', 'InputData\\navdata.tsv', '50ms', 'OutputStages\\resampledCmds.tsv',
                           'OutputStages\\resampledNav.tsv', "OutputStages\\mergedResampled.tsv")

# DataForCM
# for some reason, a column of tabs is expected at the beginning of navdata file otherwise it cannot be parsed validly
# add that column to navdataCM.tsv file
addTabs('InputData\\navdataCM.tsv', 'InputData\\navdataCMTABS.tsv')

mergedCM = ReadResampleMerge('InputData\\commandsCM.tsv', 'InputData\\navdataCMTABS.tsv', '50ms', 'OutputStages\\resampledCmdsCM.tsv',
                           'OutputStages\\resampledNavCM.tsv', "OutputStages\\mergedResampledCM.tsv")
# DataDTSecondSet
addTabs('InputData\\navdataSecondSet.tsv', 'InputData\\navdataSecondSetTABS.tsv')

mergedSecondSet = ReadResampleMerge('InputData\\cmdsSecondSet.tsv', 'InputData\\navdataSecondSetTABS.tsv', '50ms',
                                'OutputStages\\resampledCmdsSecondSet.tsv', 'OutputStages\\resampledNavSecondSet.tsv',
                                'OutputStages\\mergedResampledSecondSet.tsv')

# DataForCMSecondSet
addTabs('InputData\\navdataSecondCM.tsv', 'InputData\\navdataSecondCMTABS.tsv')

mergedSecondCM = ReadResampleMerge('InputData\\cmdsSecondCM.tsv', 'InputData\\navdataSecondCMTABS.tsv', '50ms',
                                'OutputStages\\resampledCmdsSecondCM.tsv', 'OutputStages\\resampledNavSecondCM.tsv',
                              'OutputStages\\mergedResampledSecondCM.tsv')

# DataForCMSecondSet
addTabs('InputData\\leftRightNavdata.tsv', 'InputData\\leftRightNavdataTABS.tsv')

mergedLeftRight = ReadResampleMerge('InputData\\leftRightCMDS.tsv', 'InputData\\leftRightNavdataTABS.tsv', '50ms',
                                'OutputStages\\resampledLeftRightCMDS.tsv', 'OutputStages\\resampledLeftRightNAV.tsv',
                              'OutputStages\\mergedResampledLeftRight.tsv')

mergedChanged = ReadResampleMerge('InputData\\commands.tsv', 'InputData\\navdata.tsv', '5ms', 'OutputStages\\resampledCmdsChanged.tsv',
                           'OutputStages\\resampledNavChanged.tsv', "OutputStages\\mergedResampledChanged.tsv")
mergedCMChanged = ReadResampleMerge('InputData\\commandsCM.tsv', 'InputData\\navdataCMTABS.tsv', '5ms', 'OutputStages\\resampledCmdsCMChanged.tsv',
                           'OutputStages\\resampledNavCMChanged.tsv', "OutputStages\\mergedResampledCMChanged.tsv")

