import pandas as pd
import numpy as np

cmdsColumnNames = ['time', 'leftRight', 'frontBack', 'up', 'angular']
cmds = pd.read_csv('commands.tsv', delimiter='\t', header=None, names=cmdsColumnNames)
cmds.time = pd.to_datetime(cmds.time, unit='ms')
cmds = cmds.set_index('time')
print("unique cmds time:", cmds.index.is_unique)

resampledCmds = cmds.resample('50ms').mean()
resampledCmds = resampledCmds.reindex().bfill()
resampledCmds = resampledCmds[cmds.index[0]:cmds.index[-1]]

resampledCmds.to_csv('resampledCmds.tsv', sep='\t')

navDataColumnNames = ['time', 'State', 'Battery_level', 'Magnetometer_x', 'Magnetometer_y', 'Magnetometer_z', 'Pressure',
                      'Temperature', 'Wind_speed', 'Wind_angle', 'Wind_compensation:pitch', 'Wind_compensation:roll',
                      'Pitch_y', 'Roll_x', 'Yaw_z', 'Altitude', 'Velocity_x',
                      'Velocity_y', 'Velocity_z', 'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
                      'Motor_1_power', 'Motor_2_power', 'Motor_3_power', 'Motor_4_power', 'time_board']

navData = pd.read_csv('navdata.tsv', delimiter='\t', header=None, usecols=range(1, 28), names=navDataColumnNames)
print("unique navdata time:", navData.time.is_unique)
#get rid of duplicated times in navData
navData.drop_duplicates(['time'], keep='first', inplace=True)
print("unique navdata time after .drop_duplicated call:", navData.time.is_unique)
navData.time = pd.to_datetime(navData.time, unit='ms')
navData = navData.set_index('time')

resampledNav = navData.resample('50ms').mean()
resampledNav = resampledNav.reindex().bfill()
resampledNav = resampledNav[cmds.index[0]:cmds.index[-1]]

resampledNav.to_csv('resampledNav.tsv', sep='\t')

# how inner - intersection, keeps only times that do have corresponding counterpart in second file
merged = pd.merge(resampledCmds, resampledNav, right_index=True, left_index=True, how='inner') #merge when time is index
#merged = pd.merge(cmds, navData, right_on="time", left_on="time", how='inner', indicator=True)

#TODO:How to get rid of duplicated times in navdata?
merged.to_csv("mergedResampled.tsv", sep='\t')
print("unique time right after merge:", merged.index.is_unique)



#substract minimal time from all times - get rid of miliseconds since start of unix
#minTime = merged.time.min()
#merged.time = merged.time-minTime

#to convert 'time' to datetime format
#dateIndex = pd.to_datetime(merged.time, unit='ms')

#TODO: after this data is indexed by senceless dates begginning with start of unix date - does it make any sence?
#TODO: can I somehow format string representation of values in particular columns while printing them to file by to_scv
#TODO: to print 'time' in some meaningfull format

'''
ts = pd.Series(data[:,0], times)
resampled=indexedByTime.resample('50ms').mean()
resampled=resampled.reindex().ffill()
resampled=resampled[indexedByTime.index[0]:indexedByTime.index[-1]]
'''

#resampled = indexedByTime.resample('5s').asfreq()
#resample_index = pd.date_range(start=indexedByTime.index[0], end=indexedByTime.index[-1], freq='50ms')
#dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=indexedByTime.columns)
#resampled = indexedByTime.combine_first(dummy_frame).interpolate(method='time').resample('50ms').asfreq()

print(merged.frontBack['2016-03-31 12:29:28.950'])

intIndexed = merged.reset_index()
#intIndexed.to_csv('tmp.tsv', sep='\t')
dataColumnNames = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean', 'Roll_FFT_SD',
                   'Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean', 'Pitch_FFT_SD', 'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean',
                   'Yaw_FFT_SD']

#create matrix to base machine learning on
intervalLen = 40
#prepare indeces for a new dataframe
ind = list()
indexIterator = 0
while indexIterator < intIndexed.index.size:
    ind.append(intIndexed.time.iloc[indexIterator])
    indexIterator += intervalLen
print(ind)
print(indexIterator)

#create an empty dataframe
data = pd.DataFrame(columns=dataColumnNames, index=ind)
data.index = pd.to_datetime(data.index, unit='ms')
data.index.name = 'time'

#fill the dataframe
indexIterator = 0
for index in data.index:
    #cmds
    #leftRight
    if merged.leftRight[index] <= -0.05:
        data.leftRight[index] = 1
    elif merged.leftRight[index] >= 0.05:
        data.leftRight[index] = 3
    else:
        data.leftRight[index] = 2

    #frontBack
    if merged.frontBack[index] <= -0.05:
        data.frontBack[index] = 1
    elif merged.frontBack[index] >= 0.05:
        data.frontBack[index] = 3
    else:
        data.frontBack[index] = 2

    #angular
    if merged.angular[index] <= -0.05:
        data.angular[index] = 1
    elif merged.angular[index] >= 0.05:
        data.angular[index] = 3
    else:
        data.angular[index] = 2

    #Roll
    data.Roll_Mean[index] = np.mean(merged.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    data.Roll_SD[index] = np.std(merged.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(merged.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    data.Roll_FFT_Mean[index] = np.mean(fft)
    data.Roll_FFT_SD[index] = np.std(fft)
    #Pitch
    data.Pitch_Mean[index] = np.mean(merged.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    data.Pitch_SD[index] = np.std(merged.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(merged.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    data.Pitch_FFT_Mean[index] = np.mean(fft)
    data.Pitch_FFT_SD[index] = np.std(fft)
    #Yaw
    data.Yaw_Mean[index] = np.mean(merged.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    data.Yaw_SD[index] = np.std(merged.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(merged.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    data.Yaw_FFT_Mean[index] = np.mean(fft)
    data.Yaw_FFT_SD[index] = np.std(fft)

    indexIterator += intervalLen

data.to_csv('data.tsv', sep='\t')