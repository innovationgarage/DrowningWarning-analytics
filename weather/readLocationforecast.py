import os
import h5py
import wget
import numpy as np
import pandas as pd
import datetime

def getDataForDay(forecast_time, outpath):
    #download data for the day
            hr = str(forecast_time).split(' ')[1].split(':')[0]
            outfile = 'arome_arctic_full_2_5km_{}{}{}T{}Z.nc'.format(forecast_time.year, forecast_time.month, forecast_time.day, hr)
            if os.path.exists(os.path.join(outpath, outfile)):
                print('{} already exists!'.format(outfile))
            else:
                url = "http://thredds.met.no/thredds/fileServer/aromearcticarchive/{}/{}/{}/arome_arctic_full_2_5km_{}{}{}T{}Z.nc".format(forecast_time.year, forecast_time.month, forecast_time.day, forecast_time.year, forecast_time.month, forecast_time.day, hr)
                filename = wget.download(url, out=os.path.join(outpath, outfile))
                print('{} download finished!'.format(filename))
                print(url)
                

def findNearestAvailableTimestamp(timestamp, hour_steps=3, closets='before'):
    available_hours = np.arange(0,24,hour_steps)
    for d in range(hour_steps):
        if closets == 'before':
            if (timestamp.hour - d in available_hours):
                hour = timestamp.hour - d
                date = str(timestamp.date())
            elif (timestamp.hour - d < available_hours[0]):
                hour = available_hours[-1]
                date = str(timestamp.date() - datetime.timedelta(days=1))
        elif closets == 'after':
            if (timestamp.hour + d in available_hours):
                hour = timestamp.hour + d
                date = str(timestamp.date())                
            elif timestamp.hour + d > available_hours[-1]:
                hour = available_hours[0]
                date = str(timestamp.date() + datetime.timedelta(days=1))
    new_timestamp = '{} {}:00:00'.format(date, hour)
    new_timestamp = datetime.datetime.strptime(new_timestamp, "%Y-%m-%d %H:%M:%S")
    return new_timestamp

def getDataForPeriod(forecast_start, forecast_end, hour_steps, outpath):
    fstart = datetime.datetime.strptime(forecast_start, "%Y-%m-%d %H:%M:%S")
    fend = datetime.datetime.strptime(forecast_end, "%Y-%m-%d %H:%M:%S")
    new_fstart = findNearestAvailableTimestamp(fstart, hour_steps=hour_steps, closets='before')
    new_fend = findNearestAvailableTimestamp(fend, hour_steps=hour_steps, closets='after')
    time_range = pd.date_range(new_fstart, new_fend, freq='{}H'.format(hour_steps))

    print('fstart')
    print(fstart)
    print(new_fstart)
    print('fend')
    print(fend)
    print(new_fend)
    print(time_range)

    for forecast_time in time_range:
        print(forecast_time)
        getDataForDay(forecast_time, outpath)

forecast_start = '2019-10-11 11:00:10'
forecast_end = '2019-10-11 17:00:00'
hour_steps = 3
outpath = 'data/'
getDataForPeriod(forecast_start, forecast_end, hour_steps, outpath)


#hf = h5py.File(filepath, 'r')
#print(list(hf.keys()))
#dataset = []
#for i in hf.values():
#    dataset.append(i)
    
