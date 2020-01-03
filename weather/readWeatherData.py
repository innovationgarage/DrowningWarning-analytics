import numpy as np
import pandas as pd
import pyproj
import datetime
import matplotlib.pyplot as plt
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
    
def dataset2df(dataset, use_keys, time_step, time_idx):
    tdf = pd.DataFrame()
    tdf['latitude'] = dataset[uparam].latitude.values.flatten()
    tdf['longitude'] = dataset[uparam].longitude.values.flatten()
    tdf['forecast_time'] = time_step
    for col in use_keys:
        tdf[col] = dataset[uparam].values[time_idx,0,:,:].flatten()
    return tdf

def calculateWindSpeedDirection(df, data, filepath, uparam, vparam, time_idx):
    wind_speed = mpcalc.wind_speed(data[uparam], data[vparam])
    wind_direction = mpcalc.wind_direction(data[uparam], data[vparam])
    df['wind_speed'] = wind_speed[time_idx,0,:,:].flatten()
    df['wind_direction'] = wind_direction[time_idx,0,:,:].flatten()
    dims = data[uparam].dims
    data['wind_speed'] = (dims, wind_speed)
    data['wind_direction'] = (dims, wind_direction)
    return df, data

camp = [69.961308, 18.703892]
filepath = 'data/arome_arctic_full_2_5km_20191011T09Z.nc'
fullshape = (949, 739)
use_keys = ['x_wind_gust_10m', 'y_wind_gust_10m'] #"U-momentum of gusts in 10m height"m/s, "V-momentum of gusts in 10m height"m/s
#dataset = readLocationForecast(filepath)
uparam, vparam = use_keys
dataset = xr.open_dataset(filepath)
dataset = dataset.metpy.parse_cf([uparam, vparam])
dataset[uparam].metpy.convert_units('knots')
dataset[vparam].metpy.convert_units('knots')
data_crs = dataset[uparam].metpy.cartopy_crs
lat = dataset[uparam].latitude
lon = dataset[uparam].longitude

time_steps = dataset[uparam].time.values
for time_idx, time_step in enumerate(time_steps):
    print(time_step)
    df = dataset2df(dataset, use_keys, time_step, time_idx)
    df, dataset = calculateWindSpeedDirection(df, dataset, filepath, use_keys[0], use_keys[1], time_idx)
    df.to_pickle('data/wind_{}.pkl'.format(time_step))

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