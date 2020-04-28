'''
#create matrix to base machine learning on
intervalLen = 40
#prepare indices for a new dataframe
ind = list()
indexIterator = 0
while indexIterator < intIndexed.index.size:
    ind.append(intIndexed.time.iloc[indexIterator])
    indexIterator += intervalLen
print(ind)
print(indexIterator)

#create an empty dataframe
dataForDT = pd.DataFrame(columns=dataColumnNames, index=ind)
dataForDT.index = pd.to_datetime(dataForDT.index, unit='ms')
dataForDT.index.name = 'time'

#fill the dataframe
indexIterator = 0
for index in dataForDT.index:
    #cmds
    #leftRight
    if merged.leftRight[index] <= -0.05:
        dataForDT.leftRight[index] = 1
    elif merged.leftRight[index] >= 0.05:
        dataForDT.leftRight[index] = 3
    else:
        dataForDT.leftRight[index] = 2

    #frontBack
    if merged.frontBack[index] <= -0.05:
        dataForDT.frontBack[index] = 1
    elif merged.frontBack[index] >= 0.05:
        dataForDT.frontBack[index] = 3
    else:
        dataForDT.frontBack[index] = 2

    #angular
    if merged.angular[index] <= -0.05:
        dataForDT.angular[index] = 1
    elif merged.angular[index] >= 0.05:
        dataForDT.angular[index] = 3
    else:
        dataForDT.angular[index] = 2

    #Roll
    dataForDT.Roll_Mean[index] = np.mean(merged.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    dataForDT.Roll_SD[index] = np.std(merged.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(merged.Roll_x.iloc[indexIterator:indexIterator + intervalLen])
    dataForDT.Roll_FFT_Mean[index] = np.mean(fft)
    dataForDT.Roll_FFT_SD[index] = np.std(fft)
    #Pitch
    dataForDT.Pitch_Mean[index] = np.mean(merged.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    dataForDT.Pitch_SD[index] = np.std(merged.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(merged.Pitch_y.iloc[indexIterator:indexIterator + intervalLen])
    dataForDT.Pitch_FFT_Mean[index] = np.mean(fft)
    dataForDT.Pitch_FFT_SD[index] = np.std(fft)
    #Yaw
    dataForDT.Yaw_Mean[index] = np.mean(merged.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    dataForDT.Yaw_SD[index] = np.std(merged.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    fft = np.fft.fft(merged.Yaw_z.iloc[indexIterator:indexIterator + intervalLen])
    dataForDT.Yaw_FFT_Mean[index] = np.mean(fft)
    dataForDT.Yaw_FFT_SD[index] = np.std(fft)

    indexIterator += intervalLen
'''


'''
ts = pd.Series(outputDF[:,0], times)
resampled=indexedByTime.resample('50ms').mean()
resampled=resampled.reindex().ffill()
resampled=resampled[indexedByTime.index[0]:indexedByTime.index[-1]]
'''

#resampled = indexedByTime.resample('5s').asfreq()
#resample_index = pd.date_range(start=indexedByTime.index[0], end=indexedByTime.index[-1], freq='50ms')
#dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=indexedByTime.columns)
#resampled = indexedByTime.combine_first(dummy_frame).interpolate(method='time').resample('50ms').asfreq()

#substract minimal time from all times - get rid of miliseconds since start of unix
#minTime = inputDF.time.min()
#inputDF.time = inputDF.time-minTime

#to convert 'time' to datetime format
#dateIndex = pd.to_datetime(inputDF.time, unit='ms')
