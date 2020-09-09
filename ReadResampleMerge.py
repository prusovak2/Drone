import pandas as pd
import numpy as np

def ReadCMDScvsIntoDFandResample(filename, frequency):
    '''
    reads CMD file and resamples it to given frequency
    :param filename:
    :param frequency:
    :return: dataFrame containing resampled CMDs
    '''
    cmdsColumnNames = ['time', 'leftRight', 'frontBack', 'up', 'angular']
    cmds = pd.read_csv(filename, delimiter='\t', header=None, names=cmdsColumnNames)
    cmds.time = pd.to_datetime(cmds.time, unit='ms')
    cmds = cmds.set_index('time')
    # print("unique cmds time:", cmds.index.is_unique)

    resampledCmds = cmds.resample(frequency).mean()
    resampledCmds = resampledCmds.reindex().bfill()
    resampledCmds = resampledCmds[cmds.index[0]:cmds.index[-1]]
    return resampledCmds

def ReadNAVDATAcsvIntoDFandResample(filename, frequency):
    '''
    reads NAVDATA file and resamples it to given frequency
    :param filename:
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

    resampledNav = navData.resample(frequency).mean()
    resampledNav = resampledNav.reindex().bfill()
    resampledNav = resampledNav[navData.index[0]:navData.index[-1]]
    return resampledNav


def ReadResampleMerge(cmdsFilename, navdataFilename, frequency, cmdOutputTsvFilename=None, navdataOutputTsvFilename=None,
                      mergedOutputTsvFilename=None):
    '''
    reads cmd and navdata files, resamples them to given frequency and results merges by ctime
    when output file names are given, prints corresponding outputs to .tsv files
    :param cmdsFilename:
    :param navdataFilename:
    :param frequency:
    :param cmdOutputTsvFilename:
    :param navdataOutputTsvFilename:
    :param mergedOutputTsvFilename:
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


