import pandas as pd
import numpy as np

cmdsColumnNames = ['time', 'leftRight', 'frontBack', 'up', 'angular']
cmds = pd.read_csv('commands.tsv', delimiter='\t', header=None, names=cmdsColumnNames)
#cmds = cmds.set_index('time')
print("unique cmds time:", cmds.time.is_unique)

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
#navData = navData.set_index('time')

# how inner - intersection, keeps only times that do have corresponding counterpart in second file
#merged = pd.merge(cmds, navData, right_index=True, left_index=True, how='inner') #merge when time is index
merged = pd.merge(cmds, navData, right_on="time", left_on="time", how='inner', indicator=True)

#TODO:How to get rid of duplicated times in navdata?
merged.to_csv("merged.tsv", sep='\t')
print("unique time right after merge:", merged.time.is_unique)

#substract minimal time from all times - get rid of miliseconds since start of unix
minTime = merged.time.min()
merged.time = merged.time-minTime

merged.to_csv("mergedTime.tsv", sep='\t')

#to convert 'time' to datetime format
dateIndex = pd.to_datetime(merged.time, unit='ms')

indexedByTime = merged.set_index('time')
indexedByTime.index = dateIndex
#TODO: after this data is indexed by senceless dates begginning with start of unix date - does it make any sence?
#TODO: can I somehow format string representation of values in particular columns while printing them to file by to_scv
#TODO: to print 'time' in some meaningfull format

indexedByTime.to_csv('output.tsv', sep='\t')

print("unique indexed by time:", indexedByTime.index.is_unique)

#ts = pd.Series(data[:,0], times)
resampled=indexedByTime.resample('50ms').mean()
resampled=resampled.reindex().ffill()
resampled=resampled[indexedByTime.index[0]:indexedByTime.index[-1]]

#resampled = indexedByTime.resample('5s').asfreq()

#resample_index = pd.date_range(start=indexedByTime.index[0], end=indexedByTime.index[-1], freq='50ms')
#dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=indexedByTime.columns)
#resampled = indexedByTime.combine_first(dummy_frame).interpolate(method='time').resample('50ms').asfreq()

resampled.to_csv('resampled.tsv', sep='\t')








