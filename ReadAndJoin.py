import pandas as pd

#TODO: find out what columns represent and rename them properly
cmdsColumnNames = ['time', 'cmd1', 'cmd2', 'cmd3', 'cmd4']
cmds = pd.read_csv('commands.tsv', delimiter='\t', header=None, names=cmdsColumnNames)
cmds = cmds.set_index('time')

#TODO: how the time is represented? miliseconds since when?
#cmds.index = pd.to_datetime(cmds.index)

navDataColumnNames = ['time', 'State', 'Battery level', 'Magnetometer x', 'Magnetometer y', 'Magnetometer z', 'Pressure',
                      'Temperature', 'Wind speed', 'Wind angle', 'Wind compensation: pitch', 'Wind compensation: roll',
                      'Pitch (Rotation in y)', 'Roll (Rotation in x)', 'Yaw (Rotation in z)','Altitude', 'Velocity in x',
                      'Velocity in y', 'Velocity in z', 'Acceleration in x', 'Acceleration in y', 'Acceleration in z',
                      'Motor 1 power', 'Motor 2 power', 'Motor 3 power', 'Motor 4 power', 'time board']
#TODO: what does second time (last column) mean?
navData = pd.read_csv('navdata.tsv', delimiter='\t', header=None, usecols=range(1, 28), names=navDataColumnNames)
navData = navData.set_index('time')

#TODO: how? inner - intersection, does it make sence to keep times that does not have corresponding counterpart in second file?
merged = pd.merge(cmds, navData, right_index=True, left_index=True, how='inner')

merged.to_csv('output.tsv', sep='\t')


print(cmds)
print(cmds.index)
print(cmds.columns)

print(navData)
print(navData.index)
print(navData.columns)
print(len(navDataColumnNames))

print(merged)
print(merged.index)
print(merged.columns)
