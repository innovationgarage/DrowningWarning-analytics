import numpy as np
import pyproj
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

def readWeatherData(filepath):
    fullshape = (949, 739)
    use_keys = ['x_wind_gust_10m', 'y_wind_gust_10m'] #"U-momentum of gusts in 10m height"m/s, "V-momentum of gusts in 10m height"m/s
    uparam, vparam = use_keys
    dataset = xr.open_dataset(filepath)
    dataset = dataset.metpy.parse_cf()#[uparam, vparam])
    dataset[uparam].metpy.convert_units('knots')
    dataset[vparam].metpy.convert_units('knots')
    data_crs = dataset[uparam].metpy.cartopy_crs
    lat = dataset[uparam].latitude
    lon = dataset[uparam].longitude

    wind_speed = mpcalc.wind_speed(dataset[uparam], dataset[uparam])
    wind_direction = mpcalc.wind_direction(dataset[uparam], dataset[uparam])
    dataset['wind_speed'] = xr.DataArray(wind_speed.magnitude, coords=dataset[uparam].coords, dims=dataset[uparam].dims)
    dataset['wind_speed'].attrs['units'] = wind_speed.units
    dataset['wind_direction'] = xr.DataArray(wind_direction.magnitude, coords=dataset[uparam].coords, dims=dataset[uparam].dims)
    dataset['wind_direction'].attrs['units'] = wind_direction.units
    return dataset, lon, lat

def getRelevantRegion(data, goal_lon, goal_lat, goal_param):
    #convert lat,lon to x,y using the projection of the dataset
    proj_wgs84 = pyproj.Proj("epsg:4326")
    data_crs = data[goal_param].metpy.cartopy_crs
    goal_x, goal_y = pyproj.transform(proj_wgs84, data_crs.proj4_params, goal_lon, goal_lat)
    grid_size = 2.5e3 #m
    n = 2. #how many grids to add/remove to/from the goal point to get the range
    minus_x = goal_x - n*grid_size
    plus_x = goal_x + n*grid_size    
    min_x = min(minus_x, plus_x)
    max_x = max(minus_x, plus_x)
    minus_y = goal_y - n*grid_size
    plus_y = goal_y + n*grid_size
    min_y = min(minus_y, plus_y)
    max_y = max(minus_y, plus_y)
    #select the region around the point at the right time
    region_array = data[goal_param].where((data.x>=min_x)&(data.x<=max_x)&(data.y>=min_y)&(data.y<=max_y))
    return region_array

def getNearestForecast(data, goal_lon, goal_lat, goal_time, goal_param):
    #convert lat,lon to x,y using the projection of the dataset
    proj_wgs84 = pyproj.Proj("epsg:4326")
    data_crs = data[goal_param].metpy.cartopy_crs
    goal_x, goal_y = pyproj.transform(proj_wgs84, data_crs.proj4_params, goal_lon, goal_lat)
    #select the closest grid
    weather_array = data[goal_param].sel(x=goal_x, y=goal_y, time=goal_time, method='nearest')
    return weather_array.values[0]

#def main():
boat_filepath = '../data/merged/capture_246058.txt'
weather_filepath = 'data/arome_arctic_full_2_5km_20191011T09Z.nc'
camp_location = [69.961308, 18.703892]
goal_lat = camp_location[0]
goal_lon = camp_location[0]
goal_time = '2019-10-11 11:16:40+00:00'

boat = pd.read_csv(boat_filepath)
weather, lon, lat = readWeatherData(weather_filepath)

uparam = 'wind_speed'
vparam = 'wind_direction'

boat['wind_speed'] = boat.apply(lambda row: getNearestForecast(weather, row['long'], row['lat'], row['timestamp'], uparam), axis=1)
boat['wind_direction'] = boat.apply(lambda row: getNearestForecast(weather, row['long'], row['lat'], row['timestamp'], vparam), axis=1)

# uparam_data = getRelevantRegion(weather, goal_lon, goal_lat, uparam)
# vparam_data = getRelevantRegion(weather, goal_lon, goal_lat, vparam)
# ud = uparam_data.to_dataframe()
# ud.reset_index(level=[0,1,2,3], inplace=True)
# vd = vparam_data.to_dataframe()
# vd.reset_index(level=[0,1,2,3], inplace=True)

# wind = ud
# wind[vparam] = vd[vparam]
# wind.dropna(inplace=True)

#main()
