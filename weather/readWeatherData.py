import h5py
import numpy as np
import pandas as pd
import pyproj
import datetime
import matplotlib.pyplot as plt
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

def readLocationForecast(filepath):
    hf = h5py.File(filepath, 'r')
    keys = list(hf.keys())
    dataset = {}
    i = 0
    for i, data in enumerate(hf.values()):
        dataset[keys[i]] = data
    return dataset
    
def dataset2df(dataset, use_keys, time_idx):
    tdf = pd.DataFrame()
    tdf['latitude'] = dataset['latitude'][()].flatten()
    tdf['longitude'] = dataset['longitude'][()].flatten()
    forecast_times = [datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in dataset['time'][()]]
    tdf['forecast_time'] = forecast_times[time_idx]
    for col in use_keys:
        tdf[col] = dataset[col][()][time_idx,0,:,:].flatten()
    return tdf

def calculateWindSpeedDirection(df, filepath, uparam, vparam, time_idx):
    data = xr.open_dataset(filepath)
    data = data.metpy.parse_cf([uparam, vparam])
    data[uparam].metpy.convert_units('knots')
    data[vparam].metpy.convert_units('knots')
    wind_speed = mpcalc.wind_speed(data[uparam], data[vparam])
    wind_direction = mpcalc.wind_direction(data[uparam], data[vparam])
    df['wind_speed'] = wind_speed[time_idx,0,:,:].flatten()
    df['wind_direction'] = wind_direction[time_idx,0,:,:].flatten()
    return df

camp = [69.961308, 18.703892]
filepath = 'data/arome_arctic_full_2_5km_20191011T09Z.nc'
fullshape = (949, 739)
use_keys = ['x_wind_gust_10m', 'y_wind_gust_10m'] #"U-momentum of gusts in 10m height"m/s, "V-momentum of gusts in 10m height"m/s
dataset = readLocationForecast(filepath)

for time_idx in range(len(dataset['time'][()])):
    print(dataset['time'][()][time_idx])
    df = dataset2df(dataset, use_keys, time_idx)
    df = calculateWindSpeedDirection(df, filepath, use_keys[0], use_keys[1], time_idx)
    df.to_pickle('data/wind_{}.pkl'.format(dataset['time'][()][time_idx]))

# for time_idx in range(3):
#     print(time_idx)
#     df = pd.DataFrame()
#     df = calculateWindSpeedDirection(df, filepath, use_keys[0], use_keys[1], time_idx)

# for i, ft in enumerate(df.forecast_time.unique()):
#     plt.figure()
#     print(i, ft)
#     lon = df[df.forecast_time==ft].longitude.values.reshape(fullshape)
#     lat = df[df.forecast_time==ft].latitude.values.reshape(fullshape)
#     u = df[df.forecast_time==ft].x_wind_gust_10m.values.reshape(fullshape)
#     v = df[df.forecast_time==ft].y_wind_gust_10m.values.reshape(fullshape)
#     ws = df[df.forecast_time==ft].wind_speed.values.reshape(fullshape)
#     plt.contourf(lon, lat, ws, vmin=df.wind_speed.min(), vmax=df.wind_speed.max())
#     plt.plot(camp[1], camp[0], 'xr', lw=3)
#     plt.colorbar()
#     plt.title(ft)
#     plt.savefig('windspeed_{}.png'.format(i))